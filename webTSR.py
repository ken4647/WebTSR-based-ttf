import cv2
import numpy as np
from model import *
from torchvision import transforms
from PIL import Image
import sys
from tqdm import tqdm


WEIGHT_NAME = 'weight/resnet18_sr_x2.pth'
SAMPLE_TIMES = 2
window_size = 13

if __name__ == "__main__":
    # 加载预训练的ResNet-18模型
    # model = Simple4Out((7,7), (2,2)).cuda()
    image_path = sys.argv[1]
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad((window_size//2,window_size//2,window_size//2,window_size//2),padding_mode='symmetric'),])
    
    model = WebSr4Out((2,2)).cuda()
    model.load_model(WEIGHT_NAME)
    image = Image.open(image_path).convert("L")
    image_tensor: torch.Tensor = transform(image)
    image_tensor = image_tensor.cuda().unsqueeze(0)
    size_x, size_y = image.size[0], image.size[1]
    
    print(f"SRing {image_path}, image_size={image.size}")
    
    model.eval()
    output_tensor = torch.Tensor(np.zeros((size_y*SAMPLE_TIMES, size_x*SAMPLE_TIMES), dtype="uint8")).cuda()
    with torch.no_grad():
        # TODO: Use big batch to accelebrate the process
        for x in tqdm(range(size_x)):
            for y in range(size_y):
                patch = image_tensor[:, :, y:y+window_size, x:x+window_size]
                out_p = model(patch)
                output_tensor[SAMPLE_TIMES*y:SAMPLE_TIMES*(y+1), SAMPLE_TIMES*x: SAMPLE_TIMES*(x+1)] = out_p[0,0,:,:]
    
    output_image = (output_tensor*255).cpu().detach().numpy()
    cv2.imwrite(f'output.jpg', output_image)
    
    
    

