// 実行方法
// gcc make_movie_single_qr.c -I"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\include" -L"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\lib" -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o make_movie_single_qr.exe

#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <direct.h>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ユーザー設定: 表示する画像ファイル名（同じフォルダに置く）
static const char* IMAGE_FILENAME = "ex_white_qr_normal.png";  // 例: create_white_qr.pyで生成
static const int WINDOW_WIDTH = 1920;
static const int WINDOW_HEIGHT = 1080;
static const int SWAP_INTERVAL = 1;  // 1で垂直同期

static void print_current_directory(void) {
    char cwd[MAX_PATH];
    if (_getcwd(cwd, sizeof(cwd))) {
        printf("[DEBUG] Current working directory: %s\n", cwd);
    } else {
        perror("[DEBUG] _getcwd failed");
    }
}

static GLuint load_texture(const char* filename) {
    printf("[DEBUG] Attempting to load texture: %s\n", filename);
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        printf("[ERROR] stbi_load failed for: %s\n", filename);
        return 0;
    }

    printf("[DEBUG] stbi_load success: %s (width=%d, height=%d, channels=%d)\n", filename, width, height, channels);

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
    printf("Loaded: %s (ID: %u)\n", filename, texture_id);
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

static double get_time_ms(void) {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000.0) / frequency.QuadPart;
}

int main(void) {
    print_current_directory();
    timeBeginPeriod(1);

    if (!glfwInit()) {
        fprintf(stderr, "GLFW initialization failed\n");
        return -1;
    }

    // ウィンドウ装飾なし、リサイズ不可、起動時最大化
    // glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Single QR Viewer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(SWAP_INTERVAL);

    GLuint texture = load_texture(IMAGE_FILENAME);
    if (texture == 0) {
        glfwDestroyWindow(window);
        glfwTerminate();
        timeEndPeriod(1);
        return -1;
    }

    printf("Starting render loop... (ESC to exit)\n");
    double last_log = get_time_ms();
    const double LOG_INTERVAL_MS = 5000.0;

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, 1);
        }

        draw_texture(texture);
        glfwSwapBuffers(window);
        glfwPollEvents();

        double now = get_time_ms();
        if (now - last_log > LOG_INTERVAL_MS) {
            printf("[DEBUG] Rendering...\n");
            last_log = now;
        }
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    timeEndPeriod(1);
    return 0;
}
