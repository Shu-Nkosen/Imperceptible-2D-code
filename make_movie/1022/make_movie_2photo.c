#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // 画像読み込み用（https://github.com/nothings/stb）

// ユーザー設定
#define SELECTED_IMAGE 4
#define BRIGHTNESS_INCREASE 2
#define BRIGHTNESS_DECREASE 2
#define INTERVAL 1

// 画像ベース名
const char* base_image_names[] = {
    "hocho", "kosen", "nagaoka_fireworks", "rice", "ex"
};

// 表示パターン
int frame_durations[] = {1, 1};
int num_patterns = 2;

// テクスチャID格納
GLuint normal_texture = 0;
GLuint inv_texture = 0;
GLuint orig_texture = 0;
GLuint texture_sequence[10];  // 最大10パターン

// 画像読み込み
GLuint load_texture(const char* filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    
    if (!data) {
        printf("Failed to load: %s\n", filename);
        return 0;
    }
    
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    // テクスチャパラメータ
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // データ転送
    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    
    stbi_image_free(data);
    printf("Loaded: %s (ID: %u)\n", filename, texture_id);
    
    return texture_id;
}

// テクスチャ描画
void draw_texture(GLuint texture_id) {
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

// 高精度タイマー
double get_time_ms() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000.0) / frequency.QuadPart;
}

int main() {
    // Windowsタイマー精度を1msに設定
    timeBeginPeriod(1);
    
    // GLFW初期化
    if (!glfwInit()) {
        fprintf(stderr, "GLFW initialization failed\n");
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Image Flicker (C)", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(INTERVAL);  // 垂直同期
    
    // 画像読み込み
    char filename[256];
    const char* base_name = base_image_names[SELECTED_IMAGE];
    
    snprintf(filename, sizeof(filename), "%s_b%d_d%d_normal.png", 
             base_name, BRIGHTNESS_INCREASE, BRIGHTNESS_DECREASE);
    normal_texture = load_texture(filename);
    
    snprintf(filename, sizeof(filename), "%s_b%d_d%d_inv.png", 
             base_name, BRIGHTNESS_INCREASE, BRIGHTNESS_DECREASE);
    inv_texture = load_texture(filename);
    
    snprintf(filename, sizeof(filename), "%s.png", base_name);
    orig_texture = load_texture(filename);
    
    // テクスチャシーケンス生成
    int is_normal = 1;
    int seq_len = 0;
    for (int i = 0; i < num_patterns; i++) {
        if (frame_durations[i] == 1) {
            texture_sequence[seq_len++] = is_normal ? normal_texture : inv_texture;
            is_normal = !is_normal;
        } else {
            texture_sequence[seq_len++] = orig_texture;
        }
    }
    
    printf("Starting render loop...\n");
    
    // メインループ
    int current_index = 0;
    int frame_counter = 0;
    double last_switch_time = get_time_ms();
    
    while (!glfwWindowShouldClose(window)) {
        draw_texture(texture_sequence[current_index]);
        
        double pre_swap = get_time_ms();
        glfwSwapBuffers(window);
        double post_swap = get_time_ms();
        
        glfwPollEvents();
        
        frame_counter++;
        
        if (frame_counter >= frame_durations[current_index]) {
            double switch_duration = post_swap - last_switch_time;
            double expected = frame_durations[current_index] * (1000.0 / 180.0);
            
            printf("Image %d: %.2fms (expected=%.2fms, frames=%d)\n",
                   current_index, switch_duration, expected, frame_counter);
            
            frame_counter = 0;
            current_index = (current_index + 1) % seq_len;
            last_switch_time = post_swap;
        }
    }
    
    // クリーンアップ
    glDeleteTextures(1, &normal_texture);
    glDeleteTextures(1, &inv_texture);
    glDeleteTextures(1, &orig_texture);
    
    glfwDestroyWindow(window);
    glfwTerminate();
    timeEndPeriod(1);
    
    return 0;
}