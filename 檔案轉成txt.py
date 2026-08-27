import shutil
from pathlib import Path

# 使用 Path.home() 自動定位到目前的 Desktop/IMS 資料夾
source_folder = Path.home() / "Desktop" / "IMS"
# 設定為 True：保留原始檔案，另外複製一份變成 .txt 檔（建議）
# 設定為 False：直接將原檔重新命名為 .txt 檔（原始檔案名會變更）
COPY_TO_NEW_FOLDER = True
# 複製目標資料夾（只有當 COPY_TO_NEW_FOLDER = True 時才會用到）
output_folder = Path.home() / "Desktop" / "IMS_TXT"

all_files = [f for f in source_folder.rglob("*") if f.is_file()]

print("=" * 60)
print(f"找到檔案數量：{len(all_files):,}")
print("=" * 60)

success = 0
skip = 0
failed = 0

for i, file in enumerate(all_files, start=1):
    try:
        if file.suffix.lower() == ".txt":
            skip += 1
            continue
        if not COPY_TO_NEW_FOLDER:
            # 直接更名（不保留原檔名）
            new_file = file.with_name(file.name + ".txt")
            if new_file.exists():
                print(f"[{i:,}/{len(all_files):,}] 已存在：{new_file.name}")
                skip += 1
                continue
            file.rename(new_file)
        else:
            # 複製並加副檔名（保留原檔）
            relative_path = file.relative_to(source_folder)
            new_file = output_folder / relative_path.with_name(
                relative_path.name + ".txt"
            )

            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, new_file)
        success += 1
        print(f"[{i:,}/{len(all_files):,}] 完成：{file.name}")
    except Exception as e:
        failed += 1
        print(f"[{i:,}/{len(all_files):,}] ❌ 失敗：{file}")
        print(f"   原因：{e}")


print("\n")
print("=" * 60)
print("                 處理完成")
print("=" * 60)
print(f"找到檔案：{len(all_files):,}")
print(f"成功處理：{success:,}")
print(f"跳過檔案：{skip:,}")
print(f"失敗檔案：{failed:,}")
print("=" * 60)