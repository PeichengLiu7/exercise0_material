import matplotlib.pyplot as plt

# 圆心的x轴坐标
x = 5
# 圆心的y轴坐标
y = 5
# 圆的半径
r = 4

#画一个图窗大小为5*5的图框
fig = plt.figure(figsize=(5, 5))

#画圆
circle = plt.Circle((x, y), r, color="black", fill="False")
plt.gcf().gca().add_artist(circle)

plt.axis('equal')
plt.xlim(0, 10)
plt.ylim(0, 10)

plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Circle
#
# # Get an example image
# import matplotlib.cbook as cbook
# image_file = cbook.get_sample_data('grace_hopper.png')
# img = plt.imread(image_file)
#
# # Make some example data
# x = np.random.rand(5)*img.shape[1]
# y = np.random.rand(5)*img.shape[0]
#
# # Create a figure. Equal aspect so circles look circular
# fig,ax = plt.subplots(1)
# ax.set_aspect('equal')
#
# # Show the image
# ax.imshow(img)
#
# # Now, loop through coord arrays, and create a circle at each x,y pair
# for xx,yy in zip(x,y):
#     circ = Circle((xx,yy),50)
#     ax.add_patch(circ)
#
# # Show the image
# plt.show()