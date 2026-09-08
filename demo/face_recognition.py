import sys
import io
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 中文显示配置 ==========
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("PyTorch 人脸识别系统")
print("=" * 60)

# ========== 2. 检查 GPU ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 使用设备: {device}")

# ========== 3. 加载模型 ==========
print("\n" + "=" * 60)
print("加载模型...")
print("=" * 60)

# MTCNN: 人脸检测和对齐
mtcnn = MTCNN(
    image_size=160,           # 输入图片大小
    margin=0,                 # 边距
    min_face_size=20,         # 最小人脸尺寸
    thresholds=[0.6, 0.7, 0.7],  # 检测阈值
    factor=0.709,             # 缩放因子
    post_process=True,
    device=device
)

# FaceNet: 人脸特征提取 (InceptionResnetV1)
facenet = InceptionResnetV1(
    pretrained='vggface2',    # 在 VGGFace2 数据集上预训练
    classify=False,           # 不进行分类，输出特征向量
    device=device
)
facenet.eval()

print("✅ 模型加载完成")
print(f"   - 人脸检测: MTCNN")
print(f"   - 特征提取: FaceNet (InceptionResnetV1)")

# ========== 4. 人脸检测与特征提取 ==========
def detect_face(image_path):
    """
    检测图片中的人脸并提取特征向量
    
    参数:
        image_path: 图片路径 (支持: jpg, png, bmp)
    
    返回:
        face_tensor: 人脸张量 (1, 3, 160, 160)
        bbox: 人脸框坐标 [x1, y1, x2, y2]
        prob: 检测置信度
    """
    # 读取图片
    image = Image.open(image_path)
    
    # 检测人脸（返回裁剪对齐后的人脸和张量）
    face_tensor, prob, points = mtcnn(image, return_prob=True)
    
    # 如果没有检测到人脸
    if face_tensor is None:
        print(f"⚠️ 未检测到人脸: {image_path}")
        return None, None, None
    
    # 移动到设备
    face_tensor = face_tensor.unsqueeze(0).to(device)
    
    return face_tensor, prob, points

def extract_features(face_tensor):
    """
    从人脸张量提取 512 维特征向量
    
    参数:
        face_tensor: 人脸张量 (1, 3, 160, 160)
    
    返回:
        features: 512 维特征向量 (numpy array)
    """
    with torch.no_grad():
        features = facenet(face_tensor)
        features = features.cpu().numpy().flatten()
    return features

def get_face_embedding(image_path):
    """
    完整流程: 检测人脸 + 提取特征
    
    参数:
        image_path: 图片路径
    
    返回:
        embedding: 512 维特征向量
        bbox: 人脸框
        prob: 置信度
    """
    face_tensor, prob, points = detect_face(image_path)
    
    if face_tensor is None:
        return None, None, None
    
    embedding = extract_features(face_tensor)
    return embedding, prob, points

def visualize_face_detection(image_path, save_path=None):
    """
    可视化人脸检测结果
    """
    # 读取图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测人脸
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    
    if boxes is None:
        print("⚠️ 未检测到人脸")
        return
    
    # 绘制检测框
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        x1, y1, x2, y2 = map(int, box)
        
        # 画矩形框
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 画置信度
        cv2.putText(image_rgb, f'Conf: {prob:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 画关键点（眼睛、鼻子、嘴巴）
        if points is not None:
            for j in range(5):
                cv2.circle(image_rgb, (int(points[i][j][0]), int(points[i][j][1])), 3, (255, 0, 0), -1)
    
    # 显示结果
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.title(f'人脸检测结果 ({len(boxes)} 张人脸)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 保存检测结果: {save_path}")
    
    plt.show()

# ========== 5. 人脸比对 ==========
def compare_faces(embedding1, embedding2, threshold=0.6):
    """
    比较两张人脸是否属于同一个人
    
    参数:
        embedding1: 第一张人脸的 512 维特征向量
        embedding2: 第二张人脸的 512 维特征向量
        threshold: 相似度阈值 (0.5~0.8 之间调整)
    
    返回:
        similarity: 余弦相似度 (0~1)
        is_same: 是否为同一人
    """
    # 归一化
    norm1 = embedding1 / np.linalg.norm(embedding1)
    norm2 = embedding2 / np.linalg.norm(embedding2)
    
    # 余弦相似度
    similarity = np.dot(norm1, norm2)
    
    # 判断是否为同一人
    is_same = similarity >= threshold
    
    return similarity, is_same

def face_recognition_demo(image_path1, image_path2, threshold=0.6):
    """
    完整的人脸识别演示: 两张图片比对
    """
    print("\n" + "=" * 60)
    print("人脸比对")
    print("=" * 60)
    
    # 提取特征
    print(f"📸 处理图片1: {image_path1}")
    emb1, prob1, _ = get_face_embedding(image_path1)
    
    print(f"📸 处理图片2: {image_path2}")
    emb2, prob2, _ = get_face_embedding(image_path2)
    
    if emb1 is None or emb2 is None:
        print("❌ 比对失败: 其中一张图片未检测到人脸")
        return
    
    # 比对
    similarity, is_same = compare_faces(emb1, emb2, threshold)
    
    # 输出结果
    print("\n" + "-" * 40)
    print("比对结果:")
    print(f"  余弦相似度: {similarity:.4f}")
    print(f"  相似度阈值: {threshold}")
    print(f"  🟢 是否为同一人: {'✅ 是' if is_same else '❌ 否'}")
    print("-" * 40)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    img1 = plt.imread(image_path1)
    img2 = plt.imread(image_path2)
    
    axes[0].imshow(img1)
    axes[0].set_title(f'图片1\n置信度: {prob1:.2f}' if prob1 else '图片1')
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(f'图片2\n置信度: {prob2:.2f}' if prob2 else '图片2')
    axes[1].axis('off')
    
    plt.suptitle(f'人脸比对: 相似度 {similarity:.3f} | {"✅ 同一人" if is_same else "❌ 不同人"}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ========== 6. 人脸数据库管理 ==========
class FaceDatabase:
    """人脸数据库：存储已知人脸的特征向量"""
    
    def __init__(self):
        self.embeddings = {}  # name -> embedding
        self.names = []      # 所有名字
        self.features = []   # 所有特征向量
    
    def add_face(self, name, embedding):
        """添加人脸到数据库"""
        self.embeddings[name] = embedding
        self.names.append(name)
        self.features.append(embedding)
        print(f"✅ 添加成功: {name}")
    
    def search(self, embedding, threshold=0.6):
        """
        在数据库中搜索最相似的人脸
        
        返回:
            name: 最匹配的名字
            similarity: 最高相似度
            is_match: 是否匹配成功
        """
        if len(self.features) == 0:
            return None, 0, False
        
        # 计算与所有人的相似度
        similarities = []
        for feat in self.features:
            sim, _ = compare_faces(embedding, feat, threshold)
            similarities.append(sim)
        
        # 找最高相似度
        max_sim = max(similarities)
        best_idx = similarities.index(max_sim)
        best_name = self.names[best_idx]
        
        return best_name, max_sim, max_sim >= threshold
    
    def visualize(self):
        """可视化数据库中的所有人脸"""
        if len(self.names) == 0:
            print("数据库为空")
            return
        
        fig, axes = plt.subplots(1, min(len(self.names), 8), figsize=(16, 3))
        if len(self.names) == 1:
            axes = [axes]
        
        for i, name in enumerate(self.names[:8]):
            axes[i].text(0.5, 0.5, name, ha='center', va='center', fontsize=12)
            axes[i].set_title(f'ID: {i+1}')
            axes[i].axis('off')
        
        plt.suptitle(f'人脸数据库 ({len(self.names)} 人)')
        plt.tight_layout()
        plt.show()

# ========== 7. 完整演示 ==========
def main():
    print("\n" + "=" * 60)
    print("人脸识别完整演示")
    print("=" * 60)
    print("""
    功能说明:
    1. 人脸检测: 使用 MTCNN 检测人脸
    2. 特征提取: 使用 FaceNet 提取 512 维特征向量
    3. 人脸比对: 计算余弦相似度判断是否为同一人
    4. 人脸检索: 在数据库中搜索匹配的人脸
    """)
    
    # ===== 示例1: 人脸检测可视化 =====
    print("\n【示例1】人脸检测可视化")
    print("=" * 40)
    print("请准备一张包含人脸的图片 (jpg/png)")
    
    # 生成测试图片 (如果本地没有，会用代码生成)
    test_image_path = generate_test_image()
    
    if test_image_path and os.path.exists(test_image_path):
        visualize_face_detection(test_image_path)
    
    # ===== 示例2: 人脸比对 =====
    print("\n【示例2】人脸比对")
    print("=" * 40)
    print("请准备两张人脸图片进行比对")
    
    # 这里用同一张图片演示 (实际使用请换成不同的图片)
    if test_image_path and os.path.exists(test_image_path):
        face_recognition_demo(test_image_path, test_image_path, threshold=0.6)
    
    # ===== 示例3: 构建人脸数据库 =====
    print("\n【示例3】构建人脸数据库")
    print("=" * 40)
    print("添加已知人脸到数据库")
    
    db = FaceDatabase()
    
    # 模拟添加 (实际使用请提取真实图片的特征)
    if test_image_path and os.path.exists(test_image_path):
        emb, prob, _ = get_face_embedding(test_image_path)
        if emb is not None:
            db.add_face("张三", emb)
            db.add_face("张三", emb)  # 模拟同一人的不同照片
            db.visualize()
    
    # ===== 示例4: 人脸检索 =====
    print("\n【示例4】人脸检索")
    print("=" * 40)
    if test_image_path and os.path.exists(test_image_path):
        emb, prob, _ = get_face_embedding(test_image_path)
        if emb is not None:
            name, sim, matched = db.search(emb, threshold=0.6)
            print(f"检索结果: {name}, 相似度: {sim:.4f}, 匹配: {matched}")

# ========== 8. 生成测试图片 ==========
def generate_test_image():
    """生成一个带有人脸的测试图片（如果本地没有图片）"""
    import os
    
    test_path = "test_face.jpg"
    
    # 如果已经存在，直接返回
    if os.path.exists(test_path):
        return test_path
    
    try:
        # 使用 OpenCV 生成一张简单的测试图片
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # 画一个简单的人脸 (圆形)
        cv2.circle(img, (150, 150), 80, (200, 180, 160), -1)  # 脸
        cv2.circle(img, (120, 120), 15, (0, 0, 0), -1)        # 左眼
        cv2.circle(img, (180, 120), 15, (0, 0, 0), -1)        # 右眼
        cv2.ellipse(img, (150, 160), (30, 20), 0, 0, 180, (0, 0, 0), 3)  # 嘴
        
        cv2.imwrite(test_path, img)
        print(f"✅ 生成测试图片: {test_path}")
        return test_path
    except:
        print("⚠️ 无法生成测试图片，请手动准备一张人脸图片")
        return None

# ========== 9. 命令行交互 ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch 人脸识别系统')
    parser.add_argument('--image1', type=str, help='第一张图片路径')
    parser.add_argument('--image2', type=str, help='第二张图片路径')
    parser.add_argument('--threshold', type=float, default=0.6, help='相似度阈值 (0.5~0.8)')
    parser.add_argument('--detect', type=str, help='检测单张图片中的人脸')
    
    args = parser.parse_args()
    
    if args.detect:
        # 人脸检测模式
        visualize_face_detection(args.detect)
    elif args.image1 and args.image2:
        # 人脸比对模式
        face_recognition_demo(args.image1, args.image2, args.threshold)
    else:
        # 交互模式
        print("""
        使用方式:
        1. 人脸检测: python face_recognition.py --detect image.jpg
        2. 人脸比对: python face_recognition.py --image1 person1.jpg --image2 person2.jpg
        3. 交互模式: python face_recognition.py (当前模式)
        """)
        main()