import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def kmeans_segmentation(img, k=3):
    """
    Phân vùng ảnh sử dụng thuật toán K-Means Clustering.
    Rất hiệu quả để gom nhóm các vùng màu chính (như mặt đường, vạch kẻ, lề cỏ).
    """
    # Chuyển ảnh 2D sang mảng 1D các pixel
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    # Tiêu chí hội tụ
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # K-Means
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Chuyển center về kiểu uint8 và mapping lại label về ảnh gốc
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_img = res.reshape((img.shape))
    
    return segmented_img

def color_threshold_segmentation(img):
    """
    Phân vùng bằng ngưỡng màu (HSV).
    Thường dùng để trích xuất riêng biệt vạch kẻ trắng hoặc vàng.
    """
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Ví dụ: Ngưỡng lọc màu TRẮNG (thường là vạch kẻ đường)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Áp dụng mask lên ảnh gốc
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def canny_edge_segmentation(img):
    """
    Phân vùng bằng phát hiện biên cạnh (Canny).
    Phù hợp để tìm mép đường hoặc ranh giới giữa đường và vật cản.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Tách biên Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def main():
    # Lấy đường dẫn tới thư mục data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    img_name = 'camera-noobstacle.jpg'
    img_path = os.path.join(data_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh tại: {img_path}")
        print("Vui lòng đảm bảo ảnh tồn tại trong thư mục data.")
        return

    # Đọc ảnh gốc
    img = cv2.imread(img_path)
    # OpenCV đọc hệ màu BGR, cần chuyển sang RGB để matplotlib hiển thị đúng màu
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Áp dụng các kỹ thuật phân vùng
    print("Đang xử lý K-Means Segmentation...")
    seg_kmeans = kmeans_segmentation(img_rgb, k=3)
    
    print("Đang xử lý Color Threshold (HSV)...")
    seg_color = color_threshold_segmentation(img_rgb)
    
    print("Đang xử lý Edge Detection (Canny)...")
    seg_edges = canny_edge_segmentation(img)

    # Khởi tạo cửa sổ đồ thị (Matplotlib)
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Image Segmentation Experiments - {img_name}", fontsize=16)

    # Ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # K-Means
    plt.subplot(2, 2, 2)
    plt.imshow(seg_kmeans)
    plt.title('K-Means Clustering (K=3)')
    plt.axis('off')

    # Color Threshold
    plt.subplot(2, 2, 3)
    plt.imshow(seg_color)
    plt.title('White Color Threshold (HSV)')
    plt.axis('off')

    # Edge Detection
    plt.subplot(2, 2, 4)
    plt.imshow(seg_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')

    plt.tight_layout()
    print("Hiển thị kết quả. Đóng cửa sổ để thoát.")
    plt.show()

if __name__ == '__main__':
    main()
