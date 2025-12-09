# Imperceptible-2D-code
my research


Cのセットアップ
https://qiita.com/ochx/items/01449d09777187790ee4

glfw3.dll
stb_image.h
がGL実行のために必要


gcc make_movie_2photo.c   -I"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\include"  -L"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\lib"   -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o make_movie_2photo.exe