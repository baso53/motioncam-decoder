#define FUSE_USE_VERSION 29

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
static std::map<std::string, std::string> gCache; // path → packed DNG data
static std::mutex gCacheMutex;
static const size_t kMaxCacheFrames = 10;
static std::deque<std::string> gCacheOrder; // FIFO list of cache keys

// Since every frame‐DNG is the same size, cache it once
static size_t gFrameSize = 0;
static std::mutex gFrameSizeMutex;

static std::mutex gDecoderMutex;
static std::vector<motioncam::Timestamp> gFrameList; // <-- cached frame timestamps

// builds "frame_%06d.dng"
static std::string frameName(int i)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%06d.dng", i);
    return buf;
}

// decode one frame into gCache[path]
// after writing to cache, if this is the first frame, record its size
static int load_frame(const std::string &path)
{
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        if (gCache.count(path))
            return 0;
    }

    // find zero‐based index
    int idx = -1;
    for (size_t i = 0; i < gFiles.size(); ++i)
        if (gFiles[i] == path)
        {
            idx = (int)i;
            break;
        }
    if (idx < 0)
        return -ENOENT;

    // decode raw + meta
    std::vector<uint16_t> raw;
    nlohmann::json meta;
    try
    {
        auto ts = gFrameList[idx];
        std::lock_guard<std::mutex> dlk(gDecoderMutex);
        gDecoder->loadFrame(ts, raw, meta);
    }
    catch (std::exception &e)
    {
        std::cerr << "EIO error: " << e.what() << "\n";
        return -EIO;
    }

    // pack into in‐memory DNG (exactly as before)…
    tinydngwriter::DNGImage dng;
    dng.SetSubfileType(false, false, false);
    dng.SetCompression(tinydngwriter::COMPRESSION_NONE);
    dng.SetBigEndian(false);
    dng.SetDNGVersion(1, 4, 0, 0);
    dng.SetDNGBackwardVersion(1, 3, 0, 0);

    unsigned w = meta["width"], h = meta["height"];
    dng.SetRowsPerStrip(h);
    dng.SetImageWidth(w);
    dng.SetImageLength(h);
    dng.SetImageData(
        (const unsigned char *)raw.data(),
        raw.size());
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);

    // Black/white levels, CFA, bits, matrices, etc… (same as your code)
    {
        std::vector<double> blackD = gContainerMetadata["blackLevel"];
        std::vector<uint16_t> blackU(blackD.size());
        for (size_t i = 0; i < blackD.size(); i++)
            blackU[i] = uint16_t(std::lround(blackD[i]));
        dng.SetBlackLevelRepeatDim(2, 2);
        dng.SetBlackLevel(uint32_t(blackU.size()), blackU.data());
        double whiteLevel = gContainerMetadata["whiteLevel"];
        dng.SetWhiteLevelRational(1, &whiteLevel);
    }
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
    {
        uint16_t bps[1] = {16};
        dng.SetBitsPerSample(1, bps);
    }
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
    {
        auto asShot = meta["asShotNeutral"].get<std::vector<double>>();
        dng.SetAsShotNeutral(3, asShot.data());
    }
    {
        uint32_t activeArea[4] = {0, 0, h, w};
        dng.SetActiveArea(activeArea);
    }
    if (gContainerMetadata.contains("software"))
        dng.SetSoftware(gContainerMetadata["software"].get<std::string>().c_str());
    if (gContainerMetadata.contains("orientation"))
        dng.SetOrientation(uint16_t(gContainerMetadata["orientation"].get<int>()));

    tinydngwriter::DNGWriter writer(false);
    writer.AddImage(&dng);
    std::ostringstream oss;
    std::string err;
    if (!writer.WriteToStream(oss, &err))
    {
        std::cerr << "DNG pack error: " << err << "\n";
        return -EIO;
    }

    // store in cache (rolling buffer)
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);

        // if we're at max capacity, evict the oldest
        if (gCache.size() >= kMaxCacheFrames)
        {
            const std::string &oldKey = gCacheOrder.front();
            gCacheOrder.pop_front();
            gCache.erase(oldKey);
        }

        // insert the new frame
        gCache[path] = oss.str();
        gCacheOrder.push_back(path);
    }

    // record the global frame size if first time
    {
        std::lock_guard<std::mutex> lk(gFrameSizeMutex);
        if (gFrameSize == 0)
        {
            std::lock_guard<std::mutex> ckl(gCacheMutex);
            gFrameSize = gCache[path].size();
        }
    }

    return 0;
}

//------------------------------------------------------------------------------
// ATTR: report uniform size (0 until we have it)
static int fs_getattr(const char *path, struct stat *st)
{
    std::cout << "fs_getattr";
    std::cout << path;
    std::cout << "\n";
    memset(st, 0, sizeof(*st));
    if (strcmp(path, "/") == 0)
    {
        st->st_mode = S_IFDIR | 0555;
        st->st_nlink = 2;
        return 0;
    }

    st->st_mode = S_IFREG | 0444;
    st->st_nlink = 1;
    {
        std::lock_guard<std::mutex> lk(gFrameSizeMutex);
        st->st_size = (off_t)gFrameSize;
    }
    return 0;
}

static int fs_readdir(const char *path, void *buf,
                      fuse_fill_dir_t filler,
                      off_t offset, struct fuse_file_info *fi)
{
    std::cout << "fs_readdir";
    std::cout << path;
    std::cout << "\n";
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
    std::cout << "fs_opendir";
    std::cout << path;
    std::cout << "\n";
    if (strcmp(path, "/") != 0)
        return -ENOENT;
    return 0;
}

static int fs_releasedir(const char *path, struct fuse_file_info *fi)
{
    std::cout << "fs_releasedir";
    std::cout << path;
    std::cout << "\n";
    (void)path;
    (void)fi;
    return 0;
}

// OPEN: do the heavy work once per file
static int fs_open(const char *path, struct fuse_file_info *fi)
{
    std::cout << "fs_open";
    std::cout << path;
    std::cout << "\n";
    std::string fn = path + 1;
    if (std::find(gFiles.begin(), gFiles.end(), fn) == gFiles.end())
        return -ENOENT;
    if ((fi->flags & 3) != O_RDONLY)
        return -EACCES;

    // decode if needed
    int err = load_frame(fn);
    if (err < 0)
        return err;

    // no per-file size map anymore
    return 0;
}

// READ: copy from in‐memory buffer
static int fs_read(const char *path,
                   char *buf,
                   size_t size,
                   off_t offset,
                   struct fuse_file_info *fi)
{
    std::cout << "fs_read";
    std::cout << path;
    std::cout << "\n";
    (void)fi;
    std::string fn = path + 1;
    std::lock_guard<std::mutex> lk(gCacheMutex);
    auto it = gCache.find(fn);
    if (it == gCache.end())
        return -ENOENT;

    auto &data = it->second;
    if ((size_t)offset >= data.size())
        return 0;
    size_t tocopy = std::min<size_t>(size, data.size() - (size_t)offset);
    memcpy(buf, data.data() + offset, tocopy);
    return (ssize_t)tocopy;
}

static int fs_statfs(const char *path, struct statvfs *st)
{
    std::cout << "fs_statfs";
    std::cout << path;
    std::cout << "\n";
    (void)path;
    memset(st, 0, sizeof(*st));
    st->f_bsize = 4096;
    st->f_frsize = 4096;
    st->f_blocks = 1024 * 1024;
    st->f_bfree = 0;
    st->f_bavail = 0;
    st->f_files = gFiles.size();
    st->f_ffree = 0;
    return 0;
}

static int fs_listxattr(const char *path, char *list, size_t size)
{
    std::cout << "fs_listxattr";
    std::cout << path;
    std::cout << "\n";
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

    // 2) build file list & grab metadata
    gFrameList = gDecoder->getFrames();
    gContainerMetadata = gDecoder->getContainerMetadata();
    std::cerr << "DEBUG: found " << gFrameList.size() << " frames\n";
    for (size_t i = 0; i < gFrameList.size(); ++i)
        gFiles.push_back(frameName(int(i)));

    // 3) pre‐warm the first frame so that gFrameSize is set
    if (!gFiles.empty())
    {
        if (int err = load_frame(gFiles[0]); err < 0)
        {
            std::cerr << "Failed to load first frame: " << err << "\n";
            return 1;
        }
    }

    // 4) run FUSE
    int fuse_argc = 5;
    char *fuse_argv[6];
    fuse_argv[0] = argv[0];
    fuse_argv[1] = (char *)"-f";
    fuse_argv[2] = (char *)"-o";
    fuse_argv[3] = (char *)"noappledouble,nobrowse,noappledouble,noapplexattr,rdonly";
    fuse_argv[4] = argv[2];
    fuse_argv[5] = nullptr;

    return fuse_main(fuse_argc, fuse_argv, &fs_ops, nullptr);
}