#define FUSE_USE_VERSION 26

#include <fuse.h>
#include <cstring>
#include <cerrno>
#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unistd.h>      // for W_OK, X_OK, etc
#include <sys/statvfs.h> // for struct statvfs in statfs callback

#include <motioncam/Decoder.hpp>
#include <nlohmann/json.hpp>

// tiny-dng-writer…
#define TINY_DNG_WRITER_IMPLEMENTATION
#include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

// globals
static motioncam::Decoder *gDecoder = nullptr;
static nlohmann::json gContainerMetadata;
static std::vector<std::string> gFiles;
static std::map<std::string, std::string> gCache; // path → packed DNG
static std::map<std::string, size_t> gSizes;      // path → file size
static std::mutex gCacheMutex;

// builds "frame_%06d.dng"
static std::string frameName(int i)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%06d.dng", i);
    return buf;
}

// decode one frame into gCache[path]
static int load_frame(const std::string &path)
{
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        if (gCache.count(path))
            return 0; // already loaded
    }

    // find index
    int idx = -1;
    for (size_t i = 0; i < gFiles.size(); ++i)
        if (gFiles[i] == path)
        {
            idx = int(i);
            break;
        }
    if (idx < 0)
        return -ENOENT;

    // decode raw + meta
    std::vector<uint16_t> raw;
    nlohmann::json meta;
    try
    {
        gDecoder->loadFrame(idx, raw, meta);
    }
    catch (...)
    {
        return -EIO;
    }

    // pack into in-memory DNG
    tinydngwriter::DNGImage dng;
    dng.SetBigEndian(false);
    dng.SetDNGVersion(0, 0, 4, 1);
    dng.SetDNGBackwardVersion(0, 0, 1, 1);

    unsigned w = meta["width"], h = meta["height"];
    dng.SetImageWidth(w);
    dng.SetImageLength(h);
    dng.SetImageData((const unsigned char *)raw.data(), raw.size());
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(h);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);

    // black/white level
    {
        auto black = gContainerMetadata["blackLevel"].get<std::vector<uint16_t>>();
        dng.SetBlackLevelRepeatDim(2, 2);
        dng.SetBlackLevel(4, black.data());
        double wl = gContainerMetadata["whiteLevel"];
        dng.SetWhiteLevelRational(1, &wl);
    }
    // CFA pattern
    {
        std::string sa = gContainerMetadata["sensorArrangment"];
        uint8_t cfa[4] = {0, 0, 0, 0};
        if (sa == "rggb")
        {
            cfa[0] = 0;
            cfa[1] = 1;
            cfa[2] = 1;
            cfa[3] = 2;
        }
        else if (sa == "bggr")
        {
            cfa[0] = 2;
            cfa[1] = 1;
            cfa[2] = 1;
            cfa[3] = 0;
        }
        else if (sa == "grbg")
        {
            cfa[0] = 1;
            cfa[1] = 0;
            cfa[2] = 2;
            cfa[3] = 1;
        }
        else if (sa == "gbrg")
        {
            cfa[0] = 1;
            cfa[1] = 2;
            cfa[2] = 0;
            cfa[3] = 1;
        }
        dng.SetCFAPattern(4, cfa);
        dng.SetCFALayout(1);
    }
    // bits per sample
    {
        uint16_t bps[1] = {16};
        dng.SetBitsPerSample(1, bps);
    }
    // color matrices
    {
        auto cm1 = gContainerMetadata["colorMatrix1"].get<std::vector<double>>();
        auto cm2 = gContainerMetadata["colorMatrix2"].get<std::vector<double>>();
        auto fm1 = gContainerMetadata["forwardMatrix1"].get<std::vector<double>>();
        auto fm2 = gContainerMetadata["forwardMatrix2"].get<std::vector<double>>();
        dng.SetColorMatrix1(3, cm1.data());
        dng.SetColorMatrix2(3, cm2.data());
        dng.SetForwardMatrix1(3, fm1.data());
        dng.SetForwardMatrix2(3, fm2.data());
    }
    // asShotNeutral
    {
        auto asShot = meta["asShotNeutral"].get<std::vector<double>>();
        dng.SetAsShotNeutral(3, asShot.data());
    }
    // active area
    {
        uint32_t aa[4] = {0, 0, (uint32_t)h, (uint32_t)w};
        dng.SetActiveArea(aa);
    }

    // write into stringstream
    tinydngwriter::DNGWriter writer(false);
    writer.AddImage(&dng);
    std::ostringstream oss;
    std::string err;
    if (!writer.WriteToStream(oss, &err))
    {
        std::cerr << "DNG pack error: " << err << "\n";
        return -EIO;
    }

    // store in cache
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        gCache[path] = oss.str();
    }
    return 0;
}

//------------------------------------------------------------------------------
// ATTR: report size=0 until opened
static int fs_getattr(const char *path, struct stat *st)
{
    memset(st, 0, sizeof(*st));
    if (strcmp(path, "/") == 0)
    {
        st->st_mode = S_IFDIR | 0555;
        st->st_nlink = 2;
        return 0;
    }

    std::string fn = path + 1;
    if (std::find(gFiles.begin(), gFiles.end(), fn) == gFiles.end())
        return -ENOENT;

    st->st_mode = S_IFREG | 0444;
    st->st_nlink = 1;
    auto sit = gSizes.find(fn);
    st->st_size = (sit == gSizes.end() ? 0 : sit->second);
    return 0;
}

static int fs_readdir(const char *path, void *buf,
                      fuse_fill_dir_t filler,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)offset;
    (void)fi;
    if (strcmp(path, "/") != 0)
        return -ENOENT;
    filler(buf, ".", nullptr, 0);
    filler(buf, "..", nullptr, 0);
    for (auto &f : gFiles)
        filler(buf, f.c_str(), nullptr, 0);
    return 0;
}

static int fs_opendir(const char *path, struct fuse_file_info *fi)
{
    if (strcmp(path, "/") != 0)
        return -ENOENT;
    return 0;
}

static int fs_releasedir(const char *path, struct fuse_file_info *fi)
{
    (void)path;
    (void)fi;
    return 0;
}

// OPEN: do the heavy work this one time
static int fs_open(const char *path, struct fuse_file_info *fi)
{
    std::string fn = path + 1;
    if (std::find(gFiles.begin(), gFiles.end(), fn) == gFiles.end())
        return -ENOENT;
    if ((fi->flags & 3) != O_RDONLY)
        return -EACCES;

    int err = load_frame(fn);
    if (err < 0)
        return err;

    // record true size
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        gSizes[fn] = gCache[fn].size();
    }
    return 0;
}

// READ: copy from in‐memory buffer
static int fs_read(const char *path,
                   char *buf,
                   size_t size,
                   off_t offset,
                   struct fuse_file_info *fi)
{
    (void)fi;
    std::string fn = path + 1;
    std::lock_guard<std::mutex> lk(gCacheMutex);
    auto it = gCache.find(fn);
    if (it == gCache.end())
        return -ENOENT;

    const auto &data = it->second;
    if ((size_t)offset >= data.size())
        return 0;
    size_t tocopy = std::min<size_t>(size, data.size() - (size_t)offset);
    memcpy(buf, data.data() + offset, tocopy);
    return (ssize_t)tocopy;
}

static int fs_statfs(const char *path, struct statvfs *st)
{
    (void)path;
    memset(st, 0, sizeof(*st));
    st->f_bsize = 4096;
    st->f_frsize = 4096;
    st->f_blocks = 1024 * 1024;
    st->f_bfree = 1024 * 1024;
    st->f_bavail = 1024 * 1024;
    st->f_files = gFiles.size() + 10;
    st->f_ffree = 100000;
    return 0;
}

static int fs_listxattr(const char *path, char *list, size_t size)
{
    (void)path;
    (void)list;
    (void)size;
    return 0;
}

static struct fuse_operations fs_ops = {
    .getattr = fs_getattr,
    .readdir = fs_readdir,
    .opendir = fs_opendir,
    .releasedir = fs_releasedir,
    .statfs = fs_statfs,
    .listxattr = fs_listxattr,
    .open = fs_open,
    .read = fs_read,
};

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.motioncam> <mountpoint>\n";
        return 1;
    }

    // 1) open decoder
    try
    {
        gDecoder = new motioncam::Decoder(argv[1]);
    }
    catch (std::exception &e)
    {
        std::cerr << "Decoder error: " << e.what() << "\n";
        return 1;
    }

    // fill file list & container metadata
    auto frames = gDecoder->getFrames();
    gContainerMetadata = gDecoder->getContainerMetadata();
    std::cerr << "DEBUG: found " << frames.size() << " frames\n";
    for (size_t i = 0; i < frames.size(); ++i)
        gFiles.push_back(frameName(int(i)));

    // build FUSE args (foreground + auto_cache)
    int fuse_argc = 5;
    char *fuse_argv[6];
    fuse_argv[0] = argv[0];
    fuse_argv[1] = (char *)"-f";
    fuse_argv[2] = (char *)"-o";
    fuse_argv[3] = (char *)"auto_cache";
    fuse_argv[4] = argv[2];
    fuse_argv[5] = nullptr;

    return fuse_main(fuse_argc, fuse_argv, &fs_ops, nullptr);
}