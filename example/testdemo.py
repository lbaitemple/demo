from deepnet.deepclassfy import LiNet

obj = LiNet(img="dog.jpg")
obj.eval()
print("Object is {}{}".format(obj.getName(), obj.getPercentage()))

