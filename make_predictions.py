from data_loader import *
from networks import *
from tqdm import tqdm
import config

#def make_prediction_files(input, mean_method='arithmetic'):

#   model_dir = config.model_dir

    # ---> 1. Make train prediction <----

    # train = pd.read_csv('../input/train.csv')

    # LABELS = config.labels
    #
    # label_idx = {label: i for i, label in enumerate(LABELS)}
    #train.set_index("fname")

    # train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    # skf = StratifiedKFold(n_splits=config.n_folds)

    #  predictions = np.zeros((1, 5))
    #  file_names = []
    #   for foldNum, (train_split, val_split) in enumerate(skf.split(train, train.label_idx)):
    #   val_set = train.iloc[val_split]
        #val_set = val_set.reset_index(drop=True)
        # print("Fold {0}, Val samples:{1}"
        #     .format(foldNum, len(val_set)))

        #  ckp = os.path.join(model_dir, 'model_best.%d.pth.tar' % foldNum)

        #     if input == 'logmel':
    #   fn, pred = predict_one_model_with_logmel(ckp, val_set)
        # print(pred.cpu().numpy())
        # file_names.extend(fn)
        #
        #   predictions = np.concatenate((predictions, pred.cpu().numpy()))

    #predictions = predictions[1:]
    # save_to_csv(file_names, predictions, 'train_predictions.csv')

    # ---> 2. Make test prediction <---
    # test_set = pd.read_csv('../input/test.csv')
    #
    # test_set.set_index("fname")
    # frame = test_set
    #
    # pred_list = []
    #
    # for i in range(config.n_folds):
        #ckp = config.model_dir + '/model_best.' + str(i) + '.pth.tar'
        #   if input == 'logmel':
    #   fn, pred = predict_one_model_with_logmel(ckp, frame)
        #
        #pred = pred.cpu().numpy()
        #       pred_list.append(pred)

    #  if mean_method == 'arithmetic':
    #    predictions = np.zeros_like(pred_list[0])
   #     for pred in pred_list:
  #          predictions = predictions + pred
 #       predictions = predictions / len(pred_list)
#
   #     print(predictions.shape)

  #  save_to_csv(fn, predictions, 'test_predictions.csv')

#def predict_one_model_with_logmel(checkpoint, frame):

    #print("=> loading checkpoint '{}'".format(checkpoint))
    #checkpoint = torch.load(checkpoint)

   # for i in checkpoint:
    #    print(i)
   # best_prec1 = checkpoint['best_prec1']
  #  model = run_method_by_string(config.arch)(pretrained=config.pretrain)
 #   model.load_state_dict(checkpoint['state_dict'])
#
 #   print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))
#
  #  input_frame_length = int(config.audio_duration * 1000 / config.frame_shift)
 #   stride = 20
#
  #  model.eval()

 #   prediction = torch.zeros((1, 5))
#
  #  file_names = []
 #   with torch.no_grad():
#
    #    for idx in tqdm(range(frame.shape[0])):
   #         filename = os.path.splitext(frame["fname"][idx])[0] + '.pkl'
  #          file_path = os.path.join(config.data_dir, filename)
 #           logmel = load_data(file_path)
#
  #          if logmel.shape[2] < input_frame_length:
 #               logmel = np.pad(logmel, ((0, 0), (0, 0), (0, input_frame_length - logmel.shape[2])), "constant")
#
    #        wins_data = []
   #         for j in range(0, logmel.shape[2] - input_frame_length + 1, stride):
  #              win_data = logmel[:, :, j: j + input_frame_length]
 #               wins_data.append(win_data)
#
 #           wins_data = np.array(wins_data)
#
 #           data = torch.from_numpy(wins_data).type(torch.FloatTensor)
#
   #         data = data.reshape([-1, 1, 64, 150])
  #          output = model(data)
 #           output = torch.sum(output, dim=0, keepdim=True)
#
 #           prediction = torch.cat((prediction, output), dim=0)
#
 #           file_names.append(frame["fname"][idx])
#
   # prediction = prediction[1:]
  #  return file_names, prediction

def make_a_submission_file():

    prediction = pd.read_csv(os.path.join(config.prediction_dir, 'test_predictions.csv'), header=None)
    prediction = prediction[prediction.columns[1:]].values
    test_set = pd.read_csv('../input/sample_submission.csv')
    result_path = os.path.join(config.prediction_dir, 'sbm.csv')
    top = np.array(config.labels)[np.argsort(-prediction, axis=1)[:, :1]]
    predicted_labels = [' '.join(list(x)) for x in top]
    test_set['label'] = predicted_labels
    test_set.set_index("fname", inplace=True)
    test_set[['label']].to_csv(result_path)
    print('Result saved as %s' % result_path)

def save_to_csv(files_name, prediction,file):
    df = pd.DataFrame(index=files_name, data=prediction)
    path = os.path.join(config.prediction_dir, file)
    df.to_csv(path, header=None)


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    config = Config(sampling_rate=22050,
                   audio_duration=1.5,
                   batch_size=128,
                   n_folds=5,
                   data_dir="../logmel+delta_w80_s10_m64",
                   model_dir='../model/logmel_delta_resnet101',
                   prediction_dir='../prediction/logmel_delta_resnet101',
                   arch='resnet101_logmel',
                   lr=0.01,
                   pretrain=False,
                   mixup=False,
                   epochs=100)

   # make_prediction_files(input='logmel', mean_method='arithmetic')
    make_a_submission_file()
