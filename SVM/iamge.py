# _*_ codig utf8 _*_
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

image = Image.open('a.png')
print((image.format, image.size,
       image.mode))  # Mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
# image.show()
# 转化为灰度图形
image1 = image.convert('L')
# image1.show()
# 灰度图形变为二值图像，黑白图像
img2 = image1.point(lambda i: 0 if i > 160 else 255)
# img2.show()
# 反正图形，一般只对灰度图像做
img3 = image.point(lambda i: 255 - i)
# img3.show()
# 大小缩放
img4 = image1.resize((32, 32))
# img4.show()
# 旋转
# 30：逆时针选择30度 ， expand:旋转之后图片大小是否发生变化，True,会变化，但是图片内容不会丢失，图片整体可能会变小False不会变化，但是
# 图片内容可能会丢失。fillcolor:多于出来的位置用此指定的演示填充。
img5 = image1.rotate(30, expand=True, fillcolor=None)
# img5.show()
# 转置
img6 = image1.transpose(Image.FLIP_LEFT_RIGHT)
# img6.show()
#剪切
box=(100,100,1000,200)
img7 = image.crop(box)
# img7.show()
#分裂，组合，粘贴
r,g,b,a = image.split()
g.show()
b= b.point(lambda i:i*5.5)
img8 = Image.merge(image.mode,(r,g,b,a))
img8.show()