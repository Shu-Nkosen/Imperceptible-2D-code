import os

def rename_files_with_numbers(directory):
    """
    指定したフォルダ内のファイルを番号順にリネームする。

    Parameters:
    - directory (str): ファイルをリネームするフォルダのパス。

    Returns:
    - List of tuples: 変更前と変更後のファイル名をリストで返す。
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # ファイル名をアルファベット順にソート
    renamed_files = []

    for index, file_name in enumerate(files, start=1):
        old_path = os.path.join(directory, file_name)
        ext = os.path.splitext(file_name)[1]  # ファイル拡張子を取得
        new_name = f"{index}{ext}"
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)
        renamed_files.append((file_name, new_name))
    
    return renamed_files

# 使用例
if __name__ == "__main__":
    target_directory = os.getcwd()  # カレントディレクトリを取得
    renamed = rename_files_with_numbers(target_directory)
    for old, new in renamed:
        print(f"Renamed: {old} -> {new}")
