import os
import glob

def compare_sizes():
    # Thư mục chứa log
    logs_dir = r"e:\robot-jeston\logs\logs"
    
    # Tìm tất cả các file csv
    csv_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    
    print(f"{'Tên file (Base)':<30} | {'Kích thước CSV (bytes)':<25} | {'Kích thước AVI (bytes)':<25} | {'Tỷ lệ (AVI/CSV)':<20}")
    print("-" * 105)
    
    for csv_path in sorted(csv_files):
        # Lấy tên file không có phần mở rộng
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        avi_path = os.path.join(logs_dir, f"{base_name}.avi")
        
        # Kích thước file CSV
        csv_size = os.path.getsize(csv_path)
        
        # Kiểm tra xem file AVI tương ứng có tồn tại không
        if os.path.exists(avi_path):
            avi_size = os.path.getsize(avi_path)
            ratio = avi_size / csv_size if csv_size > 0 else 0
            ratio_str = f"{ratio:.2f}x"
        else:
            avi_size = "Không có file"
            ratio_str = "N/A"
            
        print(f"{base_name:<30} | {csv_size:<25,} | {avi_size if isinstance(avi_size, str) else f'{avi_size:,}':<25} | {ratio_str:<20}")

if __name__ == "__main__":
    compare_sizes()
