import time
import numpy as np
from src import UNet
import torch
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
cuda = torch.cuda.is_available() and device.type != 'cpu'
classes = 1  # exclude background
model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)
model_path = r"./save_weights/best_model.pth"
export_path = r"./save_weights/unet_best.onnx"

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)

providers = onnxruntime.get_available_providers()
print(providers)


def export():
    inputs = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        inputs.to(device),
        export_path,
        opset_version=12,
        do_constant_folding=True,  # 启用常量折叠
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )

if __name__ == '__main__':
    export()
    inputs = torch.randn(1, 3, 640, 640).to(device)
    providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(export_path, providers=providers)
    input_name = session.get_inputs()[0].name  # 获取模型的输入名称
    output_name = session.get_outputs()[0].name  # 获取模型的输出名称
    x = Image.open(r'./data/test/image/1-0309-00ng_20240309105013211.png').convert('RGB')
    # to_tensor = transforms.ToTensor()
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(x)
    # expand batch dimension
    to_tensor = torch.unsqueeze(img, dim=0)

    # data = to_tensor(x)[None]
    data = to_tensor.numpy()
    for i in range(2):
        print("process new image")
        start_time = time.time()  # 记录开始时间
        results = session.run([output_name], {input_name: data})
        end_time = time.time()  # 记录结束时间
        execution_time = (end_time - start_time) * 1000  # 计算执行时间，并转换为毫秒
        print(f"函数执行时间: {execution_time:.2f}毫秒")
        prediction = results[0].argmax(1).squeeze(0).astype(np.uint8)

        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save(f"{i}_test_result.png")







