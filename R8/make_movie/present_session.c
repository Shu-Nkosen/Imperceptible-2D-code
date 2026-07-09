// R8 sweep presenter (GLFW)
// - Slate: black -> red (for sync detection)
// - Then 60 blocks, each block_sec long
// - In each block, alternate normal/inv textures every frame (or every repeat frame)
//
// Build example (PowerShell, run inside R8/make_movie):
//   gcc present_session.c -I"..\\..\\vcpkg\\installed\\x64-mingw-dynamic\\include" -L"..\\..\\vcpkg\\installed\\x64-mingw-dynamic\\lib" -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o present_session.exe

#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct {
    int intensity;         // e.g. 4/8/12
    const char* channel;   // "R","G","B","max","min"
    const char* token;     // file suffix token: R/G/B/X/I
    const char* base_name; // e.g. rice
} Condition;

// NOTE: keep in sync with available base images in R8/make_movie (*.png)
static const char* BASE_IMAGES[] = { "rice", "nagaoka_fireworks", "hocho", "ex" };
static const int BASE_IMAGE_COUNT = 4;
static const int INTENSITIES[] = { 4, 8, 12 };
static const int INTENSITY_COUNT = 3;
static const char* CHANNELS[] = { "R", "G", "B", "min", "max" };
static const char* TOKENS[]   = { "R", "G", "B", "I",   "X"   };
static const int CHANNEL_COUNT = 5;

static void usage(void) {
    printf("Usage: present_session.exe --rate <Hz> --exp <250|125|60> --interval <n> --out-manifest <path.json> [--block-sec 6] [--slate-sec 0.5] [--padding-sec 5] [--repeat 1]\\n");
    printf("Note: set OS display refresh to target rate before running (esp. 120Hz).\\n");
}

static int parse_int_arg(int argc, char** argv, const char* key, int default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    }
    return default_value;
}

static double parse_double_arg(int argc, char** argv, const char* key, double default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return atof(argv[i + 1]);
    }
    return default_value;
}

static const char* parse_str_arg(int argc, char** argv, const char* key, const char* default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return default_value;
}

static int has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], key) == 0) return 1;
    }
    return 0;
}

static GLuint load_texture(const char* filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        printf("[ERROR] stbi_load failed: %s\\n", filename);
        return 0;
    }

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    return texture_id;
}

static void draw_texture(GLuint texture_id) {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1,  1);
    glTexCoord2f(0, 0); glVertex2f(-1,  1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

static void draw_solid(float r, float g, float b) {
    glDisable(GL_TEXTURE_2D);
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void draw_black_frames(GLFWwindow* window, int frames) {
    for (int f = 0; f < frames && !glfwWindowShouldClose(window); f++) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
        draw_solid(0.0f, 0.0f, 0.0f);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

static int calc_frames(double sec, int refresh_hz, int interval) {
    if (sec <= 0) return 0;
    double fps = ((double)refresh_hz) / (double)interval;
    int frames = (int)(sec * fps + 0.5);
    if (frames < 1) frames = 1;
    return frames;
}

static void write_manifest(
    const char* out_path,
    int rate_hz,
    int exp,
    int interval,
    int refresh_hz,
    double block_sec,
    double slate_sec,
    double padding_sec,
    int repeat,
    Condition* conds,
    int cond_count
) {
    if (!out_path || strlen(out_path) == 0) return;

    // Create parent directories if needed.
    // Example: "manifests\\r180_e250.json"
    // This is a minimal recursive mkdir implementation for Windows paths.
    {
        char tmp[MAX_PATH];
        strncpy(tmp, out_path, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        for (char* p = tmp; *p; p++) {
            if (*p == '/' || *p == '\\') {
                char saved = *p;
                *p = '\0';
                if (strlen(tmp) > 0) {
                    CreateDirectoryA(tmp, NULL);
                }
                *p = saved;
            }
        }
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) {
        printf("[WARN] cannot write manifest: %s\\n", out_path);
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"rate_hz\": %d,\n", rate_hz);
    fprintf(f, "  \"exp\": %d,\n", exp);
    fprintf(f, "  \"interval\": %d,\n", interval);
    fprintf(f, "  \"monitor_hz\": %d,\n", refresh_hz);
    fprintf(f, "  \"block_sec\": %.6f,\n", block_sec);
    fprintf(f, "  \"slate_sec\": %.6f,\n", slate_sec);
    fprintf(f, "  \"padding_sec\": %.6f,\n", padding_sec);
    fprintf(f, "  \"repeat\": %d,\n", repeat);
    fprintf(f, "  \"conditions\": [\n");
    for (int i = 0; i < cond_count; i++) {
        Condition* c = &conds[i];
        fprintf(
            f,
            "    {\"idx\": %d, \"image\": \"%s\", \"channel\": \"%s\", \"token\": \"%s\", \"intensity\": %d}%s\n",
            i, c->base_name, c->channel, c->token, c->intensity, (i + 1 == cond_count) ? "" : ","
        );
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    printf("[OK] wrote manifest: %s\\n", out_path);
}

static void build_conditions(Condition* out, int* out_count) {
    int idx = 0;
    // Order: channel (outer) -> image (middle) -> intensity (inner)
    for (int ci = 0; ci < CHANNEL_COUNT; ci++) {
        for (int bi = 0; bi < BASE_IMAGE_COUNT; bi++) {
            for (int ii = 0; ii < INTENSITY_COUNT; ii++) {
                out[idx].base_name = BASE_IMAGES[bi];
                out[idx].channel = CHANNELS[ci];
                out[idx].token = TOKENS[ci];
                out[idx].intensity = INTENSITIES[ii];
                idx++;
            }
        }
    }
    *out_count = idx;
}

static void filename_for(char* out, size_t out_size, const Condition* c, const char* mode) {
    // mode: "normal" | "inv"
    snprintf(out, out_size, "%s_%d_%s%s.png", c->base_name, c->intensity, mode, c->token);
}

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        usage();
        return 0;
    }

    int rate_hz = parse_int_arg(argc, argv, "--rate", 0);
    int exp = parse_int_arg(argc, argv, "--exp", 0);
    int interval = parse_int_arg(argc, argv, "--interval", 1);
    int repeat = parse_int_arg(argc, argv, "--repeat", 1);
    double block_sec = parse_double_arg(argc, argv, "--block-sec", 6.0);
    double slate_sec = parse_double_arg(argc, argv, "--slate-sec", 0.5);
    double padding_sec = parse_double_arg(argc, argv, "--padding-sec", 5.0);
    const char* out_manifest = parse_str_arg(argc, argv, "--out-manifest", "");

    if (interval < 1) interval = 1;
    if (repeat < 1) repeat = 1;

    timeBeginPeriod(1);

    if (!glfwInit()) {
        fprintf(stderr, "GLFW initialization failed\\n");
        timeEndPeriod(1);
        return 1;
    }

    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    int refresh_hz = 60;
    int width = 1920;
    int height = 1080;
    if (mode) {
        refresh_hz = mode->refreshRate;
        width = mode->width;
        height = mode->height;
    }

    glfwWindowHint(GLFW_RED_BITS, mode ? mode->redBits : 8);
    glfwWindowHint(GLFW_GREEN_BITS, mode ? mode->greenBits : 8);
    glfwWindowHint(GLFW_BLUE_BITS, mode ? mode->blueBits : 8);
    glfwWindowHint(GLFW_REFRESH_RATE, refresh_hz);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(width, height, "R8 Presenter", primary, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create window\\n");
        glfwTerminate();
        timeEndPeriod(1);
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(interval);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    printf("[INFO] monitor_hz=%d interval=%d => fps~%.3f\\n", refresh_hz, interval, ((double)refresh_hz)/(double)interval);
    if (rate_hz > 0) printf("[INFO] rate_hz(arg)=%d (OS display should match)\\n", rate_hz);
    if (exp > 0) printf("[INFO] exp(arg)=%d\\n", exp);

    Condition conds[128];
    int cond_count = 0;
    build_conditions(conds, &cond_count);
    if (cond_count != 60) {
        printf("[WARN] expected 60 conditions, got %d\\n", cond_count);
    }

    write_manifest(out_manifest, rate_hz, exp, interval, refresh_hz, block_sec, slate_sec, padding_sec, repeat, conds, cond_count);

    const int padding_frames = calc_frames(padding_sec, refresh_hz, interval);
    const int slate_frames = calc_frames(slate_sec, refresh_hz, interval);
    const int block_frames = calc_frames(block_sec, refresh_hz, interval);

    // Texture cache: load all textures upfront (simple, avoids jitter)
    GLuint normal_tex[128];
    GLuint inv_tex[128];
    memset(normal_tex, 0, sizeof(normal_tex));
    memset(inv_tex, 0, sizeof(inv_tex));

    char fname[256];
    for (int i = 0; i < cond_count; i++) {
        filename_for(fname, sizeof(fname), &conds[i], "normal");
        normal_tex[i] = load_texture(fname);
        filename_for(fname, sizeof(fname), &conds[i], "inv");
        inv_tex[i] = load_texture(fname);
        if (normal_tex[i] == 0 || inv_tex[i] == 0) {
            printf("[ERROR] missing texture(s) for idx=%d. Did you run gen_assets.py?\\n", i);
            printf("        normal=%u inv=%u\\n", normal_tex[i], inv_tex[i]);
            glfwDestroyWindow(window);
            glfwTerminate();
            timeEndPeriod(1);
            return 1;
        }
    }

    printf(
        "[INFO] timeline: pre_padding=%d -> slate_black=%d -> slate_red=%d -> post_padding=%d frames\\n",
        padding_frames, slate_frames, slate_frames, padding_frames
    );

    draw_black_frames(window, padding_frames);

    // Slate: black -> red (sync signal)
    draw_black_frames(window, slate_frames);
    for (int f = 0; f < slate_frames && !glfwWindowShouldClose(window); f++) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
        draw_solid(1.0f, 0.0f, 0.0f);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    draw_black_frames(window, padding_frames);

    printf("[INFO] start conditions: %d blocks, block_frames=%d, repeat=%d\\n", cond_count, block_frames, repeat);

    for (int i = 0; i < cond_count && !glfwWindowShouldClose(window); i++) {
        GLuint t_normal = normal_tex[i];
        GLuint t_inv = inv_tex[i];
        int flip = 0;
        for (int f = 0; f < block_frames && !glfwWindowShouldClose(window); f++) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
            // alternate every 'repeat' frames
            if (repeat == 1) {
                flip = f & 1;
            } else {
                flip = ((f / repeat) & 1);
            }
            draw_texture(flip ? t_inv : t_normal);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        if ((i + 1) % 10 == 0 || (i + 1) == cond_count) {
            printf("[INFO] progress: %d/%d\\n", i + 1, cond_count);
        }
    }

    for (int i = 0; i < cond_count; i++) {
        if (normal_tex[i]) glDeleteTextures(1, &normal_tex[i]);
        if (inv_tex[i]) glDeleteTextures(1, &inv_tex[i]);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    timeEndPeriod(1);
    printf("[OK] finished\\n");
    return 0;
}

