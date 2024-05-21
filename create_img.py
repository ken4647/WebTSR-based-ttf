import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

OUTPUT_PATH = "./data"
TXT_STR = "ABCDEFGHIJKLMNOPQRSTUVWSYZabcdefghijklmnopqrstuvwsyz1234567890.*/-+--='`;./\ ,,,,,.......????????????的一是不了在有人这上大来和我个中地为他生要们以到国时就出说会也子学发着对作能可于成用过动主下而年分得家种里多经自现同后产方工行面那小所起去之都然理进体还定实如么物法你好性民从天化等力本长心把部义样事看业当因高十开些社前又它水其没想意三只重点与使但度由道全制明相两情外间二关活正合者形应头无量表象气文展"
TXT_STR += " "*10

# 创建一个空白的图像
height, width = 8192, 8192
truthground = np.zeros((height, width), dtype="uint8") + 255
background = np.zeros((height, width), dtype="uint8") + 255
# 创建渐变色的背景
# 定义渐变色的起始和结束颜色
start_color = 240 
end_color = 255
for i in range(int(height/2)):
    color = start_color * (height - i) / height + end_color * i / height,
    background[i,:] = color

grounds = [("truth",truthground), ("vari",background)]
text_list = []
begin_x, begin_y = 0, 0
for i in range(3200):
    text_attr = {}
    # 定义要添加的文字
    text_attr["text"] = "".join(random.sample(TXT_STR,random.randrange(1,20)))
    # 定义字体和大小
    font_path = "ttf/microsoftYaHei.ttf"  # 这里需要指定一个中文字体的路径
    font_size = random.randrange(8,96)
    text_attr["font"] = ImageFont.truetype(font_path, font_size)
    # 定义文字颜色为白色
    text_attr["text_color"] = random.randrange(0,224) if random.randrange(0,10) >8 else random.randrange(0,64)
    # 定义文字的放置位置
    text_attr["position"] = (begin_x, begin_y)
    begin_x += font_size*len(text_attr["text"])
    if begin_x > width:
        begin_x %= width
        begin_y += 96
    text_list.append(text_attr)

for (index,ground) in grounds:
    # 将OpenCV的图像转换为Pillow的图像
    image_pil = Image.fromarray(ground)

    # 创建一个可以在Pillow图像上绘图的对象
    draw = ImageDraw.Draw(image_pil)

    for tstr in text_list:
        draw.text(tstr["position"], tstr["text"], fill=tstr["text_color"] if index!="truth" else 0, font=tstr["font"], antialias=False)

    # 将Pillow的图像转换回OpenCV的图像
    image = np.array(image_pil)

    # 显示图像
    cv2.imwrite(f'{OUTPUT_PATH}/data_hr_{index}.bmp', image)
    cv2.imwrite(f'{OUTPUT_PATH}/data_hr_{index}.jpg', image)
    # 应用阈值处理
    _, binary_image = cv2.threshold(image, 224, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{OUTPUT_PATH}/data_trace_{index}.bmp', binary_image)
    downsampled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{OUTPUT_PATH}/data_lr_{index}.jpg', downsampled_image, [int(cv2.IMWRITE_JPEG_QUALITY),35])
