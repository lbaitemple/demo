from torchvision  import models
import torch
from PIL import Image
import pkg_resources

from torchvision import transforms

class LiNet():
    def __init__(self, img="dog.jpg"):
        self.alexnet = models.alexnet(pretrained=True)
        self.img = img
        self.classfyname = ""
        self.percentage = 0
        self.transform = transforms.Compose([            #[1]
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224),                #[3]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=[0.485, 0.456, 0.406],                #[6]
            std=[0.229, 0.224, 0.225]                  #[7]
            )])


    def getName(self):
        return self.classfyname

    def getPercentage(self):
        return self.percentage

    def eval(self):
        img = Image.open(self.img)
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        self.alexnet.eval()
        out = self.alexnet(batch_t)
        labelfile = pkg_resources.resource_filename('deepnet',
                                                    "imagenet_classes.txt")
        with open(labelfile) as f:
            labels = [line.strip() for line in f.readlines()]

        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        self.classfyname = labels[index[0]]
        self.percentage = percentage[index[0]].item()



