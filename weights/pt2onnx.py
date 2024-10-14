import torch
import onnx
import os

from models import smokeformer_v7_0_75_100_25 as create_model

def pth_to_onnx(input_path, output_path):

    #device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"using {device} device.")

    model = create_model()
    model_weight_path = input_path
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.to(device)

    checkpoint = torch.load(model_weight_path, map_location='cpu')
    # 读取checkpoint也需将模型保存到model_without_ddp
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # 输入一张224*224的三通道图像并生成张量
    x = torch.randn(1, 3, 224, 224)          
    
    # 输出.onnx文件的文件路径及文件名
    export_onnx_file = output_path          
    
    # 导出ONNX模型
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=12,    # 操作集版本，稳定操作集为9
                      do_constant_folding=True,          # 是否执行常量折叠优化
                      input_names=["input"],             # 输入名
                      output_names=["output"],           # 输出名
                      dynamic_axes={"input": {0: "batch_size"},         # 批处理变量
                                    "output": {0: "batch_size"}}
                      )

# 调用函数的示例：
pth_to_onnx('./model-best.pth', './weights/model-best.onnx')