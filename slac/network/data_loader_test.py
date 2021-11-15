import torch
import torchvision.transforms as T
import time

# to fix random seed, use torch.manual_seed
# instead of random.seed
torch.manual_seed(12)

transforms = torch.nn.Sequential(
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.3),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
scripted_transforms = torch.jit.script(transforms)
# Note: we can similarly use T.Compose to define transforms
# transforms = T.Compose([...]) and
# scripted_transforms = torch.jit.script(torch.nn.Sequential(*transforms.transforms))

tensor_image = torch.randint(0, 256, size=(1000, 3, 256, 256), dtype=torch.uint8)
# works directly on Tensors
tensor_image = tensor_image.cuda()
batched_image = torch.randint(0, 256, size=(1000, 3, 256, 256), dtype=torch.uint8).cuda()

start = time.time()
out_image1 = transforms(tensor_image)
# on the GPU
out_image1_cuda = transforms(tensor_image)
# with batches
out_image_batched = transforms(batched_image)
# and has torchscript support
out_image2 = scripted_transforms(tensor_image)
end = time.time()
print(f"Runtime of the program is {end - start}")
