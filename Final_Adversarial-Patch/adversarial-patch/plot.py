from PIL import Image
import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-l","--log_path", type=str, required=True, help="Path to log folder")
parser.add_argument("-m","--is_model_inceptionv3", default=True, help="is model: inceptionv3 or others")
args = parser.parse_args()
print(args)

def show_test_success_fail():
  img_size = 299 if args.is_model_inceptionv3 else 224
  width = img_size*5
  height = img_size*4
  new_im = Image.new('RGB', (width, height))

  img_paths = []
  row_1 = ['{}/test_output/success_{}.png'.format(args.log_path, epoch) for epoch in range(1,6)]
  row_2 = ['{}/test_output/success_{}.png'.format(args.log_path, epoch) for epoch in range(6,11)]
  row_3 = ['{}/test_output/fail_{}.png'.format(args.log_path, epoch) for epoch in range(1,6)]
  row_4 = ['{}/test_output/fail_{}.png'.format(args.log_path, epoch) for epoch in range(6,11)]
  img_paths = [row_1, row_2, row_3, row_4 ]
  images = [[Image.open(x) for x in row] for row in img_paths] 

  for i,row in enumerate(images):
    for j,im in enumerate(row):
      new_im.paste(im, (img_size*j,img_size*i))
  new_im.save('{}/success_fail.png'.format(args.log_path))

def attack_comparison():
  #img_size = 299 if args.is_model_inceptionv3 else 224
  img_size = 74  if args.is_model_inceptionv3 else 56
  width = img_size*5
  height = img_size*4
  new_im = Image.new('RGB', (width, height))

  img_paths = []
  row_1 = ['{}/comparison/before_attack_epoch{}.png'.format(args.log_path, epoch) for epoch in range(1,6)]
  row_2 = ['{}/comparison/after_attack_epoch{}.png'.format(args.log_path, epoch) for epoch in range(1,6)]
  row_3 = ['{}/comparison/before_attack_epoch{}.png'.format(args.log_path, epoch) for epoch in range(6,11)]
  row_4 = ['{}/comparison/after_attack_epoch{}.png'.format(args.log_path, epoch) for epoch in range(6,11)]
  img_paths = [row_1, row_2, row_3, row_4 ]
  images = [[Image.open(x) for x in row] for row in img_paths] 

  for i,row in enumerate(images):
    for j,im in enumerate(row):
      new_im.paste(im, (img_size*j,img_size*i))
  new_im.save('{}/comparison_attack.png'.format(args.log_path))


def patch_comparison():
  img_size = 74 if args.is_model_inceptionv3 else 56
  width = img_size*5
  height = img_size*2
  new_im = Image.new('RGB', (width, height))

  img_paths = []
  row_1 = ['{}/patch/patch_859_{}.png'.format(args.log_path, epoch) for epoch in range(1,7)]
  row_2 = ['{}/patch/patch_859_{}.png'.format(args.log_path, epoch) for epoch in range(6,11)]
  img_paths = [row_1, row_2 ]
  images = [[Image.open(x) for x in row] for row in img_paths] 

  for i,row in enumerate(images):
    for j,im in enumerate(row):
      new_im.paste(im, (img_size*j,img_size*i))
  new_im.save('{}/comparison_patch.png'.format(args.log_path))


# accuracy
def loadHistory(log_path):
    with open('{}/history.json'.format(log_path), 'r') as f:
        history = json.load(f)

    train_acc = [acc for acc in history['train']]
    test_acc  = [acc for acc in history['test']]

    return train_acc, test_acc

def plot_accuracy():
  train_acc, test_acc = loadHistory(args.log_path)

  plt.figure(figsize=(7,5))
  plt.title('Accuracy')
  plt.plot(range(1,len(train_acc)+1), train_acc, label='train')
  plt.plot(range(1,len(test_acc)+1), test_acc, label='test')
  plt.xticks(range(1,len(train_acc)+1))
  plt.legend()
  plt.show()

  plt.savefig('{p}/accuracy_{p}.png'.format(p=args.log_path))


if __name__ == "__main__":
  show_test_success_fail()
  attack_comparison()
  patch_comparison()
  plot_accuracy()
