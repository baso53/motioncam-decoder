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
#include <cstring>    // for strdup, strerror
#include <errno.h>

#ifdef _WIN32
  #include <windows.h>
  #include <shlwapi.h>    // for PathRemoveFileSpecA
  #include <fcntl.h>      // for O_RDONLY, O_BINARY, etc.
  #include <io.h>         // for _open/_read etc.
  #include <BaseTsd.h>    // for SSIZE_T
  typedef SSIZE_T ssize_t;
  #pragma comment(lib,"shlwapi.lib")
#else
  #include <unistd.h>
  #include <sys/stat.h>   // for mkdir
  #include <sys/statvfs.h>
  #include <dirent.h>     // for scanning directory
  #include <mach-o/dyld.h> // For _NSGetExecutablePath
#endif

#include <nlohmann/json.hpp>
#include <motioncam/Decoder.hpp>
#include <audiofile/AudioFile.h>

#define TINY_DNG_WRITER_IMPLEMENTATION
    #include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

bool getAudio(
    std::vector<uint8_t>& fileData,
    const int sampleRateHz,
    const int numChannels,
    std::vector<motioncam::AudioChunk>& audioChunks)
{
    AudioFile<int16_t> audio;
    audio.setNumChannels(numChannels);
    audio.setSampleRate(sampleRateHz);

    if (numChannels == 2) {
        for (auto& x : audioChunks) {
            for (auto i = 0; i < x.second.size(); i += 2) {
                audio.samples[0].push_back(x.second[i]);
                audio.samples[1].push_back(x.second[i+1]);
            }
        }
    }
    else if (numChannels == 1) {
        for (auto& x : audioChunks) {
            for (auto i = 0; i < x.second.size(); i++)
                audio.samples[0].push_back(x.second[i]);
        }
    }

    return audio.getFileData(fileData);
}

struct FSContext {
    motioncam::Decoder *decoder = nullptr;
    nlohmann::json containerMetadata;
    std::vector<std::string> filenames;
    std::map<std::string,std::string> frameCache;
    static constexpr size_t MAX_CACHE_FRAMES = 5;
    std::deque<std::string> frameCacheOrder;
    size_t frameSize = 0;
    std::vector<motioncam::Timestamp> frameList;

    std::vector<uint16_t> blackLevels;
    double whiteLevel = 0.0;
    std::array<uint8_t,4> cfa = {{0,1,1,2}};
    uint16_t orientation = 1;
    std::vector<float> colorMatrix1, colorMatrix2, forwardMatrix1, forwardMatrix2;
    std::vector<uint8_t> audioWavData;
    size_t audioSize = 0;

    std::string baseName;
};

static std::map<std::string,FSContext> contexts;

static void cache_container_metadata(FSContext *ctx)
{
    auto blackLevel = ctx->containerMetadata["blackLevel"].get<std::vector<float>>();
    ctx->blackLevels.clear();
    ctx->blackLevels.reserve(blackLevel.size());
    for (float v : blackLevel)
        ctx->blackLevels.push_back(uint16_t(std::lround(v)));

    ctx->whiteLevel = ctx->containerMetadata["whiteLevel"];
    std::string sensorArrangement =
        ctx->containerMetadata["sensorArrangment"];
    ctx->colorMatrix1 =
        ctx->containerMetadata["colorMatrix1"].get<std::vector<float>>();
    ctx->colorMatrix2 =
        ctx->containerMetadata["colorMatrix2"].get<std::vector<float>>();
    ctx->forwardMatrix1 =
        ctx->containerMetadata["forwardMatrix1"].get<std::vector<float>>();
    ctx->forwardMatrix2 =
        ctx->containerMetadata["forwardMatrix2"].get<std::vector<float>>();

    if (sensorArrangement == "rggb")
        ctx->cfa = {{0,1,1,2}};
    else if (sensorArrangement == "bggr")
        ctx->cfa = {{2,1,1,0}};
    else if (sensorArrangement == "grbg")
        ctx->cfa = {{1,0,2,1}};
    else if (sensorArrangement == "gbrg")
        ctx->cfa = {{1,2,0,1}};
    else
        ctx->cfa = {{0,1,1,2}};
}

static std::string frameName(const std::string &base, int i)
{
    char buf[1000];
    std::snprintf(buf,sizeof(buf),"%s_%06d.dng",base.c_str(),i);
    return buf;
}

static int load_frame(FSContext *ctx, const std::string &path)
{
    if (ctx->frameCache.count(path))
        return 0;

    int idx = -1;
    for (size_t i = 0; i < ctx->filenames.size(); ++i) {
        if (ctx->filenames[i] == path) {
            idx = int(i);
            break;
        }
    }
    if (idx < 0)
        return -ENOENT;

    std::vector<uint16_t> raw;
    nlohmann::json metadata;
    try {
        auto ts = ctx->frameList[idx];
        ctx->decoder->loadFrame(ts, raw, metadata);
    }
    catch (std::exception &e) {
        std::cerr << "EIO error: " << e.what() << "\n";
        return -EIO;
    }

    tinydngwriter::DNGImage dng;
    dng.SetCustomFieldLong(0x23,23);
    unsigned int width  = metadata["width"];
    unsigned int height = metadata["height"];
    auto asShotNeutral = metadata["asShotNeutral"].get<std::vector<float>>();

    dng.SetBigEndian(false);
    dng.SetDNGVersion(1,4,0,0);
    dng.SetDNGBackwardVersion(1,1,0,0);
    dng.SetImageData((const unsigned char*)raw.data(), raw.size());
    dng.SetImageWidth(width);
    dng.SetImageLength(height);
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(height);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2,2);
    dng.SetBlackLevelRepeatDim(2,2);
    dng.SetBlackLevel(uint32_t(ctx->blackLevels.size()),
                      ctx->blackLevels.data());
    dng.SetWhiteLevel(ctx->whiteLevel);
    dng.SetCompression(tinydngwriter::COMPRESSION_NONE);
    dng.SetCFAPattern(4,ctx->cfa.data());
    dng.SetCFALayout(1);
    const uint16_t bps[1] = {16};
    dng.SetBitsPerSample(1,bps);
    dng.SetColorMatrix1(3,ctx->colorMatrix1.data());
    dng.SetColorMatrix2(3,ctx->colorMatrix2.data());
    dng.SetForwardMatrix1(3,ctx->forwardMatrix1.data());
    dng.SetForwardMatrix2(3,ctx->forwardMatrix2.data());
    dng.SetAsShotNeutral(3,asShotNeutral.data());
    dng.SetCalibrationIlluminant1(21);
    dng.SetCalibrationIlluminant2(17);
    dng.SetUniqueCameraModel("MotionCam");
    dng.SetSubfileType();
    const uint32_t activeArea[4] = {0,0,height,width};
    dng.SetActiveArea(&activeArea[0]);
    if (ctx->orientation)
        dng.SetOrientation(ctx->orientation);

    tinydngwriter::DNGWriter writer(false);
    writer.AddImage(&dng);
    std::ostringstream oss;
    std::string err;
    if (!writer.WriteToFile(oss,&err)) {
        std::cerr << "DNG pack error: " << err << "\n";
        return -EIO;
    }

    if (ctx->frameCache.size() >= FSContext::MAX_CACHE_FRAMES) {
        ctx->frameCache.erase(ctx->frameCacheOrder.front());
        ctx->frameCacheOrder.pop_front();
    }
    ctx->frameCache[path] = oss.str();
    ctx->frameCacheOrder.push_back(path);

    if (ctx->frameSize == 0)
        ctx->frameSize = ctx->frameCache[path].size();

    return 0;
}

// ─── FUSE callbacks ────────────────────

// 1) getattr: use struct fuse_stat instead of struct stat
static int fs_getattr(const char *path, struct fuse_stat *st)
{
    std::string p(path);
    memset(st,0,sizeof(*st));

    if (p == "/") {
        st->st_mode = S_IFDIR | 0555;
        st->st_nlink = 2;
        return 0;
    }

    std::string rest = p.substr(1);
    auto slash = rest.find('/');
    if (slash == std::string::npos) {
        if (contexts.count(rest)) {
            st->st_mode = S_IFDIR | 0555;
            st->st_nlink = 2;
            return 0;
        }
        return -ENOENT;
    }

    std::string base  = rest.substr(0,slash);
    std::string fname = rest.substr(slash+1);
    auto it = contexts.find(base);
    if (it == contexts.end()) return -ENOENT;
    FSContext &ctx = it->second;

    std::string audioName = ctx.baseName + ".wav";
    if (fname == audioName) {
        if (ctx.audioSize == 0) return -ENOENT;
        st->st_mode = S_IFREG | 0444;
        st->st_nlink = 1;
        st->st_size  = (off_t)ctx.audioSize;
        return 0;
    }

    if (std::find(ctx.filenames.begin(),
                  ctx.filenames.end(),
                  fname) == ctx.filenames.end())
        return -ENOENT;

    st->st_mode = S_IFREG | 0444;
    st->st_nlink = 1;
    st->st_size  = (off_t)ctx.frameSize;
    return 0;
}

// 2) readdir: use fuse_off_t for offset
static int fs_readdir(const char *path,
                      void *buf,
                      fuse_fill_dir_t filler,
                      fuse_off_t offset,
                      struct fuse_file_info *fi)
{
    std::string p(path);
    (void)offset; (void)fi;

    if (p == "/") {
        filler(buf,".",nullptr,0);
        filler(buf,"..",nullptr,0);
        for (auto &kv : contexts)
            filler(buf,kv.first.c_str(),nullptr,0);
        return 0;
    }

    std::string rest = p.substr(1);
    auto it = contexts.find(rest);
    if (it == contexts.end()) return -ENOENT;
    FSContext &ctx = it->second;

    filler(buf,".",nullptr,0);
    filler(buf,"..",nullptr,0);

    for (auto &f : ctx.filenames)
        filler(buf,f.c_str(),nullptr,0);

    if (ctx.audioSize) {
        std::string audioName = ctx.baseName + ".wav";
        filler(buf,audioName.c_str(),nullptr,0);
    }
    return 0;
}

static int fs_open(const char *path, struct fuse_file_info *fi)
{
    std::string p(path);
    if (p.size()<2 || p[0]!='/') return -ENOENT;
    std::string rest = p.substr(1);
    auto slash = rest.find('/');
    if (slash == std::string::npos) return -EISDIR;

    std::string base  = rest.substr(0,slash);
    std::string fname = rest.substr(slash+1);
    auto it = contexts.find(base);
    if (it == contexts.end()) return -ENOENT;
    FSContext &ctx = it->second;

    std::string audioName = ctx.baseName + ".wav";
    if (fname == audioName) {
        return ((fi->flags & 3) == O_RDONLY)
               ? 0
               : -EACCES;
    }

    if (std::find(ctx.filenames.begin(),
                  ctx.filenames.end(),
                  fname) == ctx.filenames.end())
        return -ENOENT;

    return ((fi->flags & 3) == O_RDONLY)
           ? 0
           : -EACCES;
}

// 3) read: use fuse_off_t for offset
static int fs_read(const char *path,
                   char *buf,
                   size_t size,
                   fuse_off_t offset,
                   struct fuse_file_info *fi)
{
    (void)fi;
    std::string p(path);
    std::string rest = p.substr(1);
    auto slash = rest.find('/');
    if (slash == std::string::npos) return -EISDIR;

    std::string base  = rest.substr(0,slash);
    std::string fname = rest.substr(slash+1);
    auto it = contexts.find(base);
    if (it == contexts.end()) return -ENOENT;
    FSContext &ctx = it->second;

    std::string audioName = ctx.baseName + ".wav";
    if (fname == audioName) {
        if ((size_t)offset >= ctx.audioSize) return 0;
        size_t tocopy =
          std::min<size_t>(size, ctx.audioSize - (size_t)offset);
        memcpy(buf, ctx.audioWavData.data() + offset, tocopy);
        return (ssize_t)tocopy;
    }

    int err = load_frame(&ctx,fname);
    if (err < 0) return err;
    auto it2 = ctx.frameCache.find(fname);
    if (it2 == ctx.frameCache.end()) return -ENOENT;
    const std::string &data = it2->second;
    if ((size_t)offset >= data.size()) return 0;
    size_t tocopy =
      std::min<size_t>(size, data.size() - (size_t)offset);
    memcpy(buf, data.data() + offset, tocopy);
    return (ssize_t)tocopy;
}

static struct fuse_operations fs_ops = {
    .getattr = fs_getattr,
    .open    = fs_open,
    .read    = fs_read,
    .readdir = fs_readdir,
};

int main(int argc, char *argv[])
{
    if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << "\n";
        return 1;
    }

    // 1) figure out our executable directory
    std::string appDir;
#ifdef _WIN32
    {
        char modulePath[MAX_PATH];
        DWORD n = GetModuleFileNameA(NULL,modulePath,MAX_PATH);
        if (n == 0 || n == MAX_PATH) {
            std::cerr<<"GetModuleFileName failed\n";
            return 1;
        }
        PathRemoveFileSpecA(modulePath);
        appDir = modulePath;
    }
#else
    {
        char exePath[PATH_MAX];
        uint32_t size = sizeof(exePath);
        if (_NSGetExecutablePath(exePath,&size) != 0) {
            std::cerr<<"Executable path too long\n";
            return 1;
        }
        appDir = dirname(exePath);
    }
#endif

    // 2) scan directory for *.mcraw
#ifdef _WIN32
    {
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA((appDir + "\\*.mcraw").c_str(),&fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                std::string fn = fd.cFileName;
                if (fn.size()>6 && fn.compare(fn.size()-6,6,".mcraw")==0) {
                    std::string fullPath = appDir + "\\" + fn;
                    std::string baseName = fn.substr(0,fn.size()-6);
                    std::cout<<"Found file: "<<fullPath<<"\n";

                    FSContext ctx;
                    ctx.baseName = baseName;
                    try {
                        ctx.decoder = new motioncam::Decoder(fullPath);
                    }
                    catch (std::exception &e) {
                        std::cerr<<"Decoder error ("<<fullPath<<"): "
                                 <<e.what()<<"\n";
                        continue;
                    }

                    ctx.frameList         = ctx.decoder->getFrames();
                    ctx.containerMetadata = ctx.decoder->getContainerMetadata();
                    cache_container_metadata(&ctx);

                    std::cerr<<"DEBUG: ["<<fullPath<<"] found "
                             <<ctx.frameList.size()<<" frames\n";

                    for (size_t i = 0; i < ctx.frameList.size(); ++i)
                        ctx.filenames.push_back(frameName(baseName,int(i)));

                    if (!ctx.filenames.empty())
                        load_frame(&ctx,ctx.filenames[0]);

                    try {
                        std::vector<motioncam::AudioChunk> audioChunks;
                        std::vector<uint8_t> fileData;
                        ctx.decoder->loadAudio(audioChunks);

                        int sampleRate  = ctx.decoder->audioSampleRateHz();
                        int numChannels = ctx.decoder->numAudioChannels();

                        getAudio(fileData,sampleRate,numChannels,audioChunks);
                        ctx.audioWavData.assign(
                          fileData.begin(),fileData.end());
                        ctx.audioSize = ctx.audioWavData.size();
                    }
                    catch (std::exception &e) {
                        std::cerr<<"Audio processing error ("
                                 <<fullPath<<"): "<<e.what()<<"\n";
                    }

                    contexts.emplace(baseName,std::move(ctx));
                }
            } while (FindNextFileA(h,&fd));
            FindClose(h);
        }
    }
#else
    {
        DIR *d = opendir(appDir.c_str());
        if (!d) {
            std::cerr<<"Error: cannot open directory "
                     <<appDir<<"\n";
            return 1;
        }
        struct dirent *ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string fn = ent->d_name;
            if (fn.size()>6 && fn.compare(fn.size()-6,6,".mcraw")==0) {
                std::string fullPath = appDir + "/" + fn;
                std::string baseName = fn.substr(0,fn.size()-6);
                std::cout<<"Found file: "<<fullPath<<"\n";

                FSContext ctx;
                ctx.baseName = baseName;
                try {
                    ctx.decoder = new motioncam::Decoder(fullPath);
                }
                catch (std::exception &e) {
                    std::cerr<<"Decoder error ("<<fullPath<<"): "
                             <<e.what()<<"\n";
                    continue;
                }

                ctx.frameList         = ctx.decoder->getFrames();
                ctx.containerMetadata = ctx.decoder->getContainerMetadata();
                cache_container_metadata(&ctx);

                std::cerr<<"DEBUG: ["<<fullPath<<"] found "
                         <<ctx.frameList.size()<<" frames\n";

                for (size_t i=0; i<ctx.frameList.size(); ++i)
                    ctx.filenames.push_back(frameName(baseName,int(i)));

                if (!ctx.filenames.empty())
                    load_frame(&ctx,ctx.filenames[0]);

                try {
                    std::vector<motioncam::AudioChunk> audioChunks;
                    std::vector<uint8_t> fileData;
                    ctx.decoder->loadAudio(audioChunks);

                    int sampleRate  = ctx.decoder->audioSampleRateHz();
                    int numChannels = ctx.decoder->numAudioChannels();

                    getAudio(fileData,sampleRate,numChannels,audioChunks);
                    ctx.audioWavData.assign(
                      fileData.begin(),fileData.end());
                    ctx.audioSize = ctx.audioWavData.size();
                }
                catch (std::exception &e) {
                    std::cerr<<"Audio processing error ("
                             <<fullPath<<"): "<<e.what()<<"\n";
                }

                contexts.emplace(baseName,std::move(ctx));
            }
        }
        closedir(d);
    }
#endif

    if (contexts.empty()) {
        std::cerr<<"No .mcraw files found in "<<appDir<<"\n";
        return 1;
    }

    // 3) ensure the mount-point exists
    std::string mountPoint = appDir
#ifdef _WIN32
      + "\\mcraws";
#else
      + "/mcraws";
    if (::mkdir(mountPoint.c_str(),0755)!=0 && errno!=EEXIST) {
        std::cerr<<"Error creating mountpoint '"<<mountPoint<<"': "
                 <<strerror(errno)<<"\n";
        return 1;
    }
#endif

    // 4) assemble FUSE args
    std::string volname =
      mountPoint.substr(mountPoint.find_last_of("/\\")+1);
    std::string mountOptions = "ro,volname=" + volname;

    int fuse_argc = 6;
    char *fuse_argv[7];
    fuse_argv[0] = argv[0];
    fuse_argv[1] = (char*)"-f";  // foreground
    fuse_argv[2] = (char*)"-s";  // single-threaded
    fuse_argv[3] = (char*)"-o";  // mount options
    fuse_argv[4] = (char*)mountOptions.c_str();
    fuse_argv[5] = (char*)mountPoint.c_str();
    fuse_argv[6] = nullptr;

    // 5) run FUSE
    int ret = fuse_main(fuse_argc,fuse_argv,&fs_ops,nullptr);
    std::cout<<"Exit code: "<<ret<<"\n";

    // cleanup mountpoint dir
#ifdef _WIN32
    RemoveDirectoryA(mountPoint.c_str());
#else
    ::rmdir(mountPoint.c_str());
#endif

    return ret;
}