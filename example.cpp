/*
 * Copyright 2023 MotionCam
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define FUSE_USE_VERSION 29

#include <fuse.h>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <sys/statvfs.h>
#include <cstring>    // for strdup, strerror
#include <libgen.h>   // for dirname(), basename()
#include <sys/stat.h> // for mkdir
#include <errno.h>

#include <motioncam/Decoder.hpp>
#include <audiofile/AudioFile.h>

#define TINY_DNG_WRITER_IMPLEMENTATION
    #include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

void writeAudio(
    const std::string& outputPath,
    const int sampleRateHz,
    const int numChannels,
    std::vector<motioncam::AudioChunk>& audioChunks)
{
    AudioFile<int16_t> audio;
    
    audio.setNumChannels(numChannels);
    audio.setSampleRate(sampleRateHz);
    
    if(numChannels == 2) {
        for(auto& x : audioChunks) {
            for(auto i = 0; i < x.second.size(); i+=2) {
                audio.samples[0].push_back(x.second[i]);
                audio.samples[1].push_back(x.second[i+1]);
            }
        }
    }
    else if(numChannels == 1) {
        for(auto& x : audioChunks) {
            for(auto i = 0; i < x.second.size(); i++)
                audio.samples[0].push_back(x.second[i]);
        }
    }
    
    audio.save(outputPath);
}

struct FSContext {
    motioncam::Decoder *decoder = nullptr;
    nlohmann::json containerMetadata;
    std::vector<std::string> filenames;
    std::map<std::string, std::string> frameCache;
    static constexpr size_t MAX_CACHE_FRAMES = 10;
    std::deque<std::string> frameCacheOrder;
    size_t frameSize = 0;
    std::vector<motioncam::Timestamp> frameList;

    std::vector<uint16_t> blackLevels;
    double whiteLevel = 0.0;
    std::array<uint8_t, 4> cfa = {{0, 1, 1, 2}};
    uint16_t orientation;
    std::vector<float> colorMatrix1,
        colorMatrix2,
        forwardMatrix1,
        forwardMatrix2;
};

// call this once, right after containerMetadata is set:
static void cache_container_metadata(FSContext *ctx)
{
    // Black levels
    std::vector<uint16_t> blackLevel = ctx->containerMetadata["blackLevel"];
    ctx->blackLevels.clear();
    ctx->blackLevels.reserve(blackLevel.size());
    for (float v : blackLevel)
        ctx->blackLevels.push_back(uint16_t(std::lround(v)));

    // White level
    ctx->whiteLevel = ctx->containerMetadata["whiteLevel"];

    // CFA pattern
    std::string sensorArrangement = ctx->containerMetadata["sensorArrangment"];
    ctx->colorMatrix1 = ctx->containerMetadata["colorMatrix1"].get<std::vector<float>>();
    ctx->colorMatrix2 = ctx->containerMetadata["colorMatrix2"].get<std::vector<float>>();
    ctx->forwardMatrix1 = ctx->containerMetadata["forwardMatrix1"].get<std::vector<float>>();
    ctx->forwardMatrix2 = ctx->containerMetadata["forwardMatrix2"].get<std::vector<float>>();

    if (sensorArrangement == "rggb")
        ctx->cfa = {{0, 1, 1, 2}};
    else if (sensorArrangement == "bggr")
        ctx->cfa = {{2, 1, 1, 0}};
    else if (sensorArrangement == "grbg")
        ctx->cfa = {{1, 0, 2, 1}};
    else if (sensorArrangement == "gbrg")
        ctx->cfa = {{1, 2, 0, 1}};
    else
        ctx->cfa = {{0, 1, 1, 2}};

    if (ctx->containerMetadata.contains("orientation"))
    {
        ctx->orientation = uint16_t(ctx->containerMetadata["orientation"].get<int>());
    }
}

static std::string frameName(int i)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%06d.dng", i);
    return buf;
}

// decode one frame into frameCache[path]
// after writing to cache, if this is the first frame, record its size
static int load_frame(const std::string &path, FSContext *ctx)
{
    { // fast‐path if cached
        if (ctx->frameCache.count(path))
            return 0;
    }

    // find the frame index
    int idx = -1;
    for (size_t i = 0; i < ctx->filenames.size(); ++i)
        if (ctx->filenames[i] == path)
        {
            idx = int(i);
            break;
        }
    if (idx < 0)
        return -ENOENT;

    // decode raw + per‐frame metadata
    std::vector<uint16_t> raw;
    nlohmann::json metadata;
    try
    {
        auto ts = ctx->frameList[idx];
        ctx->decoder->loadFrame(ts, raw, metadata);
    }
    catch (std::exception &e)
    {
        std::cerr << "EIO error: " << e.what() << "\n";
        return -EIO;
    }

    // pack into a DNGImage
    tinydngwriter::DNGImage dng;
    const unsigned int width = metadata["width"];
    const unsigned int height = metadata["height"];
    std::vector<float> asShotNeutral = metadata["asShotNeutral"];
    dng.SetBigEndian(false);
    dng.SetDNGVersion(1, 4, 0, 0);
    dng.SetDNGBackwardVersion(1, 1, 0, 0);
    dng.SetImageData(
        (const unsigned char *)raw.data(),
        raw.size());
    dng.SetImageWidth(width);
    dng.SetImageLength(height);
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(height);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);
    
    dng.SetBlackLevelRepeatDim(2, 2);
    dng.SetBlackLevel(uint32_t(ctx->blackLevels.size()), ctx->blackLevels.data());
    dng.SetWhiteLevel(ctx->whiteLevel);
    dng.SetCompression(tinydngwriter::COMPRESSION_NONE);

    dng.SetCFAPattern(4, ctx->cfa.data());
    
    // Rectangular
    dng.SetCFALayout(1);

    const uint16_t bps[1] = { 16 };
    dng.SetBitsPerSample(1, bps);
    
    dng.SetColorMatrix1(3, ctx->colorMatrix1.data());
    dng.SetColorMatrix2(3, ctx->colorMatrix2.data());

    dng.SetForwardMatrix1(3, ctx->forwardMatrix1.data());
    dng.SetForwardMatrix2(3, ctx->forwardMatrix2.data());
    
    dng.SetAsShotNeutral(3, asShotNeutral.data());
    
    dng.SetCalibrationIlluminant1(21);
    dng.SetCalibrationIlluminant2(17);
    
    dng.SetUniqueCameraModel("MotionCam");
    dng.SetSubfileType();
    
    const uint32_t activeArea[4] = { 0, 0, height, width };
    dng.SetActiveArea(&activeArea[0]);
    if (ctx->orientation) {
        dng.SetOrientation(ctx->orientation);
    }

    // Write DNG
    std::string err;
    tinydngwriter::DNGWriter writer(false);
    writer.AddImage(&dng);
    std::ostringstream oss;
    if (!writer.WriteToFile(oss, &err))
    {
        std::cerr << "DNG pack error: " << err << "\n";
        return -EIO;
    }

    // insert into rolling‐buffer cache
    {
        if (ctx->frameCache.size() >= FSContext::MAX_CACHE_FRAMES)
        {
            ctx->frameCache.erase(ctx->frameCacheOrder.front());
            ctx->frameCacheOrder.pop_front();
        }
        ctx->frameCache[path] = oss.str();
        ctx->frameCacheOrder.push_back(path);
    }

    // record frame‐size once
    {
        if (ctx->frameSize == 0)
        {
            ctx->frameSize = ctx->frameCache[path].size();
        }
    }

    return 0;
}

// report uniform size (0 until we have it)
static int fs_getattr(const char *path, struct stat *st)
{
    FSContext *ctx = static_cast<FSContext*>(fuse_get_context()->private_data);
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
        st->st_size = (off_t)ctx->frameSize;
    }
    return 0;
}

static int fs_readdir(const char *path, void *buf,
                      fuse_fill_dir_t filler,
                      off_t offset, struct fuse_file_info *fi)
{
    FSContext *ctx = static_cast<FSContext*>(fuse_get_context()->private_data);
    std::cout << "fs_readdir";
    std::cout << path;
    std::cout << "\n";
    (void)offset;
    (void)fi;
    if (strcmp(path, "/") != 0)
        return -ENOENT;
    filler(buf, ".", nullptr, 0);
    filler(buf, "..", nullptr, 0);
    for (auto &f : ctx->filenames)
        filler(buf, f.c_str(), nullptr, 0);
    return 0;
}

static int fs_open(const char *path, struct fuse_file_info *fi)
{
    FSContext *ctx = static_cast<FSContext*>(fuse_get_context()->private_data);
    std::cout << "fs_open" << path << "\n";
    std::string fn = path + 1;
    if (std::find(ctx->filenames.begin(), ctx->filenames.end(), fn) == ctx->filenames.end())
        return -ENOENT;
    if ((fi->flags & 3) != O_RDONLY)
        return -EACCES;

    return 0;
}

// do the expensive decoding here, once per file
static int fs_read(const char *path,
                   char *buf,
                   size_t size,
                   off_t offset,
                   struct fuse_file_info *fi)
{
    FSContext *ctx = static_cast<FSContext*>(fuse_get_context()->private_data);
    (void)fi;
    std::cout << "fs_read" << path << "\n";

    std::string fn = path + 1;

    // 1) trigger the lazy‐decode if we haven't already cached it
    int err = load_frame(fn, ctx);
    if (err < 0)
        return err;

    // 2) we know it's in cache now; copy out the bytes

    auto it = ctx->frameCache.find(fn);
    if (it == ctx->frameCache.end())
        return -ENOENT; // should never happen, load_frame just inserted it

    const std::string &data = it->second;
    if ((size_t)offset >= data.size())
        return 0;

    size_t tocopy = std::min<size_t>(size, data.size() - (size_t)offset);
    memcpy(buf, data.data() + offset, tocopy);
    return (ssize_t)tocopy;
}

static struct fuse_operations fs_ops = {
    .getattr = fs_getattr,
    .readdir = fs_readdir,
    .open = fs_open,
    .read = fs_read,
};

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.motioncam>\n";
        return 1;
    }

    FSContext ctx;

    // 1) figure out mount‐point directory: same folder, same basename (no .mcraw)
    std::string inputPath = argv[1];
    // make writable copies for dirname()/basename()
    char *dup1 = strdup(inputPath.c_str());
    char *dup2 = strdup(inputPath.c_str());

    std::string parentDir = dirname(dup1); // e.g. "/foo/bar"
    std::string fileName = basename(dup2); // e.g. "video.mcraw"

    free(dup1);
    free(dup2);

    // strip extension from filename
    std::string base = fileName;
    auto dot = base.rfind('.');
    if (dot != std::string::npos)
        base.erase(dot);

    // assemble mount‐point path
    std::string mountPoint = parentDir + "/" + base;

    // create the directory if it doesn't exist
    if (::mkdir(mountPoint.c_str(), 0755) != 0 && errno != EEXIST)
    {
        std::cerr << "Error creating mountpoint '" << mountPoint
                  << "': " << strerror(errno) << "\n";
        return 1;
    }

    // 2) open decoder
    try
    {
        ctx.decoder = new motioncam::Decoder(inputPath);
    }
    catch (std::exception &e)
    {
        std::cerr << "Decoder error: " << e.what() << "\n";
        return 1;
    }

    // 3) preload metadata & frame‐list
    ctx.frameList = ctx.decoder->getFrames();
    ctx.containerMetadata = ctx.decoder->getContainerMetadata();
    cache_container_metadata(&ctx);

    std::cerr << "DEBUG: found " << ctx.frameList.size() << " frames\n";
    for (size_t i = 0; i < ctx.frameList.size(); ++i)
        ctx.filenames.push_back(frameName(int(i)));

    // 4) warm up first frame so frameSize is known
    if (!ctx.filenames.empty())
    {
        if (int err = load_frame(ctx.filenames[0], &ctx); err < 0)
        {
            std::cerr << "Failed to load first frame: " << err << "\n";
            return 1;
        }
    }

    // 5) build FUSE argv and run
    int fuse_argc = 6;
    char *fuse_argv[7];
    fuse_argv[0] = argv[0];
    fuse_argv[1] = (char *)"-f";
    fuse_argv[2] = (char *)"-s";
    fuse_argv[3] = (char *)"-o";
    std::string last = mountPoint.substr(mountPoint.find_last_of('/') + 1);
    std::string mountOptions = "iosize=8388608,noappledouble,nobrowse,rdonly,noapplexattr,volname=" + last;
    fuse_argv[4] = (char *)mountOptions.c_str();
    // finally, our auto‐created mountpoint:
    fuse_argv[5] = const_cast<char *>(mountPoint.c_str());
    fuse_argv[6] = nullptr;

    return fuse_main(fuse_argc, fuse_argv, &fs_ops, &ctx);
}