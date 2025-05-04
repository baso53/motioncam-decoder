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

#include <motioncam/Decoder.hpp>
#include <nlohmann/json.hpp>

#define TINY_DNG_WRITER_IMPLEMENTATION
#include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

static motioncam::Decoder *gDecoder = nullptr;
static nlohmann::json gContainerMetadata;
static std::vector<std::string> gFiles;           // "frame_000000.dng", ...
static std::map<std::string, std::string> gCache; // filename -> in‐memory DNG
static std::mutex gCacheMutex;

// helper: generate the filename for frame index i
static std::string frameName(int i)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%06d.dng", i);
    return std::string(buf);
}

// lazy‐load a frame into gCache[path]
static int load_frame(const std::string &path)
{
    // already in cache?
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        if (gCache.count(path))
            return 0;
    }
    // find index
    int idx = -1;
    for (size_t i = 0; i < gFiles.size(); ++i)
    {
        if (gFiles[i] == path)
        {
            idx = static_cast<int>(i);
            break;
        }
    }
    if (idx < 0)
        return -ENOENT;

    // decode + pack to DNG in memory
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

    // fill DNGImage
    tinydngwriter::DNGImage dng;
    dng.SetBigEndian(false);
    dng.SetDNGVersion(0, 0, 4, 1);
    dng.SetDNGBackwardVersion(0, 0, 1, 1);
    unsigned w = meta["width"], h = meta["height"];
    dng.SetImageWidth(w);
    dng.SetImageLength(h);
    dng.SetImageData(reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(h);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);
    // ... copy blackLevel, whiteLevel, CFAPattern etc as you had before ...
    // For brevity assume container metadata fields are correct:
    {
        auto black = gContainerMetadata["blackLevel"].get<std::vector<uint16_t>>();
        dng.SetBlackLevelRepeatDim(2, 2);
        dng.SetBlackLevel(4, black.data());
        double wl = gContainerMetadata["whiteLevel"];
        dng.SetWhiteLevelRational(1, &wl);
    }
    std::string sa = gContainerMetadata["sensorArrangment"];
    std::vector<uint8_t> cfa(4);
    if (sa == "rggb")
        cfa = {0, 1, 1, 2};
    else if (sa == "bggr")
        cfa = {2, 1, 1, 0};
    else if (sa == "grbg")
        cfa = {1, 0, 2, 1};
    else if (sa == "gbrg")
        cfa = {1, 2, 0, 1};
    dng.SetCFAPattern(4, cfa.data());
    dng.SetCFALayout(1);
    const uint16_t bps[1] = {16};
    dng.SetBitsPerSample(1, bps);
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
    auto asShot = meta["asShotNeutral"].get<std::vector<double>>();
    dng.SetAsShotNeutral(3, asShot.data());
    uint32_t aa[4] = {0, 0, static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
    dng.SetActiveArea(aa);

    // writer
    tinydngwriter::DNGWriter writer(false);
    writer.AddImage(&dng);
    std::ostringstream oss;
    std::string err;
    if (!writer.WriteToStream(oss, &err))
    {
        std::cerr << "DNG pack error: " << err;
        return -EIO;
    }

    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        gCache[path] = oss.str();
    }
    return 0;
}

// FUSE callbacks:

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
    auto it = std::find(gFiles.begin(), gFiles.end(), fn);
    if (it == gFiles.end())
        return -ENOENT;
    // ensure we know the size (lazy load):
    if (load_frame(fn))
        return -EIO;
    size_t sz;
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        sz = gCache[fn].size();
    }
    st->st_mode = S_IFREG | 0444;
    st->st_nlink = 1;
    st->st_size = sz;
    return 0;
}

static int fs_readdir(const char *path, void *buf,
                      fuse_fill_dir_t filler,
                      off_t offset, struct fuse_file_info *fi)
{
    (void)offset;
    (void)fi;
    if (strcmp(path, "/"))
        return -ENOENT;
    filler(buf, ".", nullptr, 0);
    filler(buf, "..", nullptr, 0);
    for (auto &f : gFiles)
        filler(buf, f.c_str(), nullptr, 0);
    return 0;
}

static int fs_open(const char *path, struct fuse_file_info *fi)
{
    std::string fn = path + 1;
    if (std::find(gFiles.begin(), gFiles.end(), fn) == gFiles.end())
        return -ENOENT;
    // read‐only only:
    if ((fi->flags & 3) != O_RDONLY)
        return -EACCES;
    // lazy load:
    int e = load_frame(fn);
    return e;
}

static int fs_read(const char *path, char *buf,
                   size_t size, off_t offset,
                   struct fuse_file_info *fi)
{
    std::string fn = path + 1;
    {
        std::lock_guard<std::mutex> lk(gCacheMutex);
        auto it = gCache.find(fn);
        if (it == gCache.end())
            return -ENOENT;
        const std::string &data = it->second;
        if (offset < (off_t)data.size())
        {
            size_t off = static_cast<size_t>(offset < 0 ? 0 : offset);
            size_t tocopy = std::min<size_t>(size, data.size() - off);
            // size_t tocopy = std::min(size, data.size() - offset);
            memcpy(buf, data.data() + offset, tocopy);
            return static_cast<int>(tocopy);
        }
        return 0;
    }
}

static struct fuse_operations fs_ops = {
    .getattr = fs_getattr,
    .readdir = fs_readdir,
    .open = fs_open,
    .read = fs_read,
};

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input.motioncam> <mountpoint>\n";
        return 1;
    }
    // 1) open decoder
    try
    {
        gDecoder = new motioncam::Decoder(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Decoder error: " << e.what() << "\n";
        return 1;
    }

    // 2) build file list
    auto frames = gDecoder->getFrames();
    gContainerMetadata = gDecoder->getContainerMetadata();
    std::cerr << "DEBUG: found " << frames.size() << " frames\n";
    for (size_t i = 0; i < frames.size(); ++i)
    {
        gFiles.push_back(frameName(static_cast<int>(i)));
    }

    // 3) shift FUSE args so argv[1]==mountpoint
    int fuse_argc = 2;
    char *fuse_argv[3];
    fuse_argv[0] = argv[0]; // your program
    fuse_argv[1] = argv[2]; // the mountpoint
    fuse_argv[2] = nullptr;

    // 4) run FUSE
    return fuse_main(fuse_argc, fuse_argv, &fs_ops, nullptr);
}