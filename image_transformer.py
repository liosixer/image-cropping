import cv2
import numpy as np
import os

class ImageTransformer:
    def __init__(self):
        # 清理debug目录
        self.debug_dir = "debug"
        if os.path.exists(self.debug_dir):
            for file in os.listdir(self.debug_dir):
                os.remove(os.path.join(self.debug_dir, file))

    def save_debug_image(self, image, name):
        """保存调试图像"""
        os.makedirs(self.debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.debug_dir, name), image)

    def detect_photos(self, image):
        """检测并分割出每一张老照片"""
        # 保存原图用于调试
        self.save_debug_image(image, "1_original.jpg")
        
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.save_debug_image(gray, "2_gray.jpg")
        
        # 使用自适应直方图均衡化增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        self.save_debug_image(enhanced, "3_enhanced.jpg")
        
        # 使用双边滤波保留边缘的同时去除噪点
        blur = cv2.bilateralFilter(enhanced, 9, 145, 145)
        self.save_debug_image(blur, "4_blur.jpg")
        
        # 使用Sobel算子进行边缘检测
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel)
        self.save_debug_image(sobel, "5_sobel.jpg")
        
        # 使用Canny边缘检测，调整阈值
        edges = cv2.Canny(sobel, 10, 100)
        self.save_debug_image(edges, "6_edges.jpg")
        
        # 形态学操作，连接边缘
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        self.save_debug_image(dilated, "7_dilated.jpg")
        
        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"找到 {len(contours)} 个轮廓")
        
        # 在原图上绘制所有轮廓
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        self.save_debug_image(contour_img, "8_all_contours.jpg")
        
        # 获取图片总面积
        total_area = image.shape[0] * image.shape[1]
        min_area = total_area * 0.05  # 最小面积阈值为总面积的5%
        
        photo_corners = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            # 获取最小外接矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 在调试图像上标记这个矩形
            rect_img = image.copy()
            cv2.drawContours(rect_img, [box], 0, (0, 0, 255), 2)
            self.save_debug_image(rect_img, f"9_rectangle_{i+1}.jpg")
            
            print(f"轮廓 {i}: 面积={area}, 占总面积比例={area/total_area:.2%}")
            photo_corners.append(box)
            
        print(f"检测到 {len(photo_corners)} 个符合条件的照片")
        return photo_corners

    def order_points(self, pts):
        """将四个角点排序为左上、右上、右下、左下"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # 计算左上角和右下角
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 计算右上角和左下角
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def perspective_transform(self, image, pts):
        """透视变换，校正为矩形"""
        # 对输入点进行排序
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # 计算新图片的宽度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # 计算新图片的高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # 构建目标点
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # 应用透视变换，使用INTER_LINEAR插值方法
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), 
                                   flags=cv2.INTER_LINEAR)
        
        return warped

    def detect_content_area(self, image):
        """检测照片中的实际内容区域，去除白边"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Otsu阈值法进行二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.save_debug_image(binary, "8_content_binary.jpg")
        
        # 形态学操作，去除噪点
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        self.save_debug_image(binary, "9_content_morph.jpg")
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的轮廓（假设这是照片的内容区域）
        if not contours:
            return None
        
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 向内收缩一点，确保完全去除边框
        margin = 2
        return (x + margin, y + margin, w - 2*margin, h - 2*margin)

    def process_image(self, input_path, output_dir):
        """处理输入图像并保存结果"""
        # 读取输入图像
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("无法读取输入图像")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取输入文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # 检测照片
        photo_corners = self.detect_photos(image)
        
        # 如果没有检测到有效的照片区域，直接输出原图
        if not photo_corners:
            print("未检测到照片区域，将直接输出原图")
            output_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            cv2.imwrite(output_path, image)
            print(f"已保存原图: {output_path}")
            return 1
        
        # 处理每个检测到的照片
        for idx, corners in enumerate(photo_corners):
            print(f"处理第 {idx+1} 张照片")
            
            # 1. 透视变换
            warped = self.perspective_transform(image, corners)
            self.save_debug_image(warped, f"10_warped_{idx+1}.jpg")
            
            # 2. 检测内容区域
            content_rect = self.detect_content_area(warped)
            if content_rect is None:
                print(f"警告：无法检测到照片 {idx+1} 的内容区域，将使用完整区域")
                content = warped  # 使用完整的透视变换结果
            else:
                # 3. 裁剪内容区域
                x, y, w, h = content_rect
                content = warped[y:y+h, x:x+w]
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{base_name}_{idx+1}.jpg")
            cv2.imwrite(output_path, content)
            print(f"已保存: {output_path}")
        
        return len(photo_corners)

def main():
    # Create an instance of the ImageTransformer class
    transformer = ImageTransformer()
    
    # Process the image
    # image_file = './input/multiple.png'
    # num_photos = transformer.process_image(image_file, 'output')
    # print(f"Processed {num_photos} photos from {image_file}")

    # return
    # Set the input and output paths
    input_dir = "./input"  # Change this to your input directory
    output_dir = "output"  # This is the output directory
    
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Process each image file
    for image_file in image_files:
        try:
            # Construct full file path
            input_path = os.path.join(input_dir, image_file)
            
            # Process the image
            num_photos = transformer.process_image(input_path, output_dir)
            print(f"Processed {num_photos} photos from {image_file}")
        except Exception as e:
            print(f"An error occurred while processing {image_file}: {str(e)}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main() 