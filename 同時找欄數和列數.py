from pathlib import Path

# 指向剛剛轉檔產生的 IMS_TXT 資料夾
folder = Path.home() / "Desktop" / "IMS_TXT"
# 自動掃描 IMS_TXT 底下所有子資料夾（1st_test, 2nd_test, 3rd_test）的 TXT 檔
txt_files = list(folder.rglob("*.txt"))
print(f"從 IMS_TXT 中共找到 {len(txt_files):,} 個 TXT 檔案\n")

total_rows = 0
success = 0
failed = 0
for file in txt_files:
    try:
        rows = 0
        column_counts = set()
        with open(file, "rb") as f:
            for line in f:
                rows += 1
                line = line.rstrip(b"\r\n")
                columns = len(line.split(b"\t"))
                column_counts.add(columns)
        total_rows += rows
        success += 1
        # 顯示相對路徑（例如：1st_test/檔案名稱.txt）
        relative_name = file.relative_to(folder)
        if len(column_counts) == 1:
            columns = next(iter(column_counts))
            print(f"{relative_name} → {rows:,} 列 × {columns} 欄")
        else:
            print(
                f"{relative_name} → "
                f"{rows:,} 列 × "
                f"欄數不一致：{sorted(column_counts)}"
            )
    except Exception as e:
        failed += 1
        print(f" {file.name} → 讀取失敗：{e}")

print("\n================================")
print("             統計結果")
print("================================")
print(f"TXT 檔案數：{len(txt_files):,}")
print(f"成功檔案數：{success:,}")
print(f"失敗檔案數：{failed:,}")
print(f"所有 TXT 總列數：{total_rows:,}")
print("================================")