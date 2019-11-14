from PIL import Image

path=r'func_vis.png'
img=Image.open(path)
shape=img.size
img=img.resize((shape[0]//2,shape[1]//2), Image.BILINEAR)
img.save('fuc_vis_bilinear.png')