import os
from PIL import Image

def clean_image_dataset(data_dir):
    """
    Duyệt qua các thư mục con trong data_dir, kiểm tra từng file ảnh.
    Nếu file bị hỏng hoặc không phải là ảnh, nó sẽ bị xóa.

    Args:
        data_dir (str): Đường dẫn đến thư mục chứa các lớp ảnh (ví dụ: ./PetImages).
    """
    print(f"Bắt đầu quá trình dọn dẹp trong thư mục: {data_dir}")
    deleted_files_count = 0
    
    # Lấy danh sách các thư mục con (mỗi thư mục là một lớp)
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    if not subfolders:
        print(f"Lỗi: Không tìm thấy thư mục con nào trong '{data_dir}'.")
        return

    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        print(f"\n--- Đang kiểm tra thư mục: {folder_name} ---")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 1. Mở file ảnh
                img = Image.open(file_path)
                # 2. Kiểm tra tính toàn vẹn của file ảnh
                img.verify()
            except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
                print(f"-> Phát hiện file lỗi: {file_path}")
                print(f"   Lý do: {e}")
                
                try:
                    # 3. Nếu có lỗi, xóa file
                    os.remove(file_path)
                    print(f"   => Đã xóa file thành công.")
                    deleted_files_count += 1
                except OSError as remove_error:
                    print(f"   Lỗi: Không thể xóa file. {remove_error}")

    print("\n-----------------------------------------")
    print("✅ Quá trình dọn dẹp hoàn tất!")
    print(f"Tổng số file lỗi đã bị xóa: {deleted_files_count}")
    print("-----------------------------------------")


# --- CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    # ⚠️ THAY ĐỔI ĐƯỜNG DẪN NÀY cho đúng với thư mục của bạn
    DATA_DIR = "F:/Data Analysic/Convolutional-Neural-Networks/PetImages" 
    
    # Gọi hàm để bắt đầu dọn dẹp
    clean_image_dataset(DATA_DIR)