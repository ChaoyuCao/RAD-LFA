<<<<<<< HEAD
import torch.optim as optim
import tqdm
import numpy as np
import torch
import os
from modules.dyformer import build_model
from data_processing.dataset import build_dataset
from parsing import parse_args
from sklearn.metrics import r2_score
import copy
import datetime
from data_processing.utils import check_dir
import random
import math
import time
import pandas as pd


def write_log_head(log_file,  args):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S\n")
    log_file.write(formatted_time)
    log_file.write('\n')
    log_file.write('Config\n')
    config_line = ''
    #* Log the arguments
    log_file.write('Experiment Configuration:')
    for arg, value in vars(args).items():
        log_file.write(f'{arg}: {value}')
    
    log_file.write(f'{config_line}\n')
    log_file.write('\n')
    
    
def write_log(log_file, performance):
    line = ''
    for key, value in performance.items():
        line += f'{key} - {value} '
    log_file.write(f'{line}\n')
    log_file.write('\n')
    return line


def calculate_accuracy(preds, labels, thresh=4 / 12.0):
    preds = np.array(preds)
    labels = np.array(labels)
    #! Check
    preds_class = np.where(preds >= thresh, 1, 0)
    labels_class = np.where(labels >= thresh, 1, 0)
    correct_count = np.sum(preds_class == labels_class)
    accuracy = correct_count / len(labels_class)
    return accuracy * 100


def save_results_to_excel(data, file_name, sheet_name, mode='w'):
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with pd.ExcelWriter(file_name, mode=mode, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name=sheet_name)


def train(args):
    # experiment log file
    log_file = open(os.path.join(f'{args.log_dir}/{args.expname}', f'log.txt'), 'w')
    write_log_head(log_file, args)
    # build dataset
    train_set, val_set, test_set = build_dataset(dataset_dir=args.dataset_dir,
                                                 series=args.series, batch_size=args.batch_size,
                                                 startpoint=args.startpoint, endpoint=args.endpoint, steps=args.steps,
                                                 resize_shape=args.resize_shape,
                                                 shuffle=True, split=args.split)

    # build model
    model = build_model(args)

    model_device = next(model.parameters()).device
    print(f'Training on {model_device}')
    print('=' * 20)
    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=args.step_size,
                                                gamma=args.gamma)
    loss_fn = torch.nn.MSELoss()
    start_epoch = 1

    best_model = None
    best_r2 = -99999.0
    best_data = {'train_gt': [], 'train_pred': [],
                 'val_gt': [], 'val_pred': [],
                 'test_gt': [], 'test_pred': []}
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # train
        train_loss = 0
        train_pred = []
        train_gt = []
        train_sample_num = 0
        model.train()
        model.train_ = True
        for batch in tqdm.tqdm(train_set):
            inputs, t, targets = batch
            inputs = inputs.to(model_device)
            targets = targets.to(model_device)
            t = t.to(model_device)
            batch_size = inputs.shape[0]
            outputs = model(inputs, t)
            optimizer.zero_grad()
            loss = loss_fn(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_pred.extend(outputs.cpu().detach().numpy().flatten())
            train_gt.extend(targets.cpu().detach().numpy().flatten())
            train_loss += loss.item() * batch_size
            train_sample_num += batch_size
        train_loss /= train_sample_num
        train_r2 = r2_score(y_true=train_gt, y_pred=train_pred)
        train_accuracy = calculate_accuracy(train_pred, train_gt, args.threshold)

        # validation
        val_loss = 0
        val_sample_num = 0
        val_pred = []
        val_gt = []
        model.eval()
        model.train_ = False
        with torch.no_grad():
            for batch in tqdm.tqdm(val_set):
                inputs, t, targets = batch
                inputs = inputs.to(model_device)
                targets = targets.to(model_device)
                t = t.to(model_device)
                batch_size = inputs.shape[0]
                outputs = model(inputs, t)
                loss = loss_fn(outputs.squeeze(), targets.squeeze())

                val_pred.extend(outputs.cpu().detach().numpy().flatten().tolist())
                val_gt.extend(targets.cpu().detach().numpy().flatten().tolist())
                val_loss += loss.item() * batch_size
                val_sample_num += batch_size

        # print(val_pred, val_gt)
        val_loss /= val_sample_num
        val_r2 = r2_score(y_true=val_gt, y_pred=val_pred)
        val_accuracy = calculate_accuracy(val_pred, val_gt, args.threshold)

        performance = {'Epoch': epoch, 'Train RMSE': math.sqrt(train_loss), 'Train R2': train_r2,
                       'Val RMSE': math.sqrt(val_loss), 'Val R2': val_r2, 'Train Acc': train_accuracy,
                       'Val Acc': val_accuracy}
        line = write_log(log_file, performance)
        print(line)

        if val_r2 > best_r2:
            best_model = copy.deepcopy(model)
            best_r2 = val_r2
            best_data['train_gt'] = train_gt
            best_data['train_pred'] = train_pred
            best_data['val_gt'] = val_gt
            best_data['val_pred'] = val_pred

            excel_file_path = os.path.join(args.log_dir, f'{args.expname}_results.xlsx')
            train_results = pd.DataFrame({
                'Train Predictions': best_data['train_pred'],
                'Train Ground Truth': best_data['train_gt']
            })
            val_results = pd.DataFrame({
                'Validation Predictions': best_data['val_pred'],
                'Validation Ground Truth': best_data['val_gt']
            })
            save_results_to_excel(train_results, excel_file_path, 'Train Set')
            save_results_to_excel(val_results, excel_file_path, 'Validation Set', mode='a')

        print('=' * 20)
    test_loss, test_r2, test_gt, test_pred, test_accuracy = test(best_model, test_set, args)
    performance['Test RMSE'] = math.sqrt(test_loss)
    performance['Test R2'] = test_r2
    performance['Test Acc'] = test_accuracy
    log_file.write('Test set performance\n')
    line = write_log(log_file, performance)
    print('Test')
    print(line)

    best_data['test_gt'] = test_gt
    best_data['test_pred'] = test_pred
    checkpoint_data = {
        'model': best_model.state_dict(),
        'val_r2': best_r2,
        'test_r2': test_r2,
        'results': best_data,
    }
    torch.save(checkpoint_data, os.path.join(args.ckp_dir, f'{args.expname}/best_model.pth'))

    #time record
    end_time = time.time()
    total_time = end_time - start_time
    log_file.write(f'Total training time: {total_time} seconds\n')

    # GPU consumed
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_usage = torch.cuda.memory_allocated() / gpu_memory
        log_file.write(f'GPU memory usage: {gpu_usage * 100:.2f}%\n')

    log_file.close()

    return checkpoint_data


def test(model, dataset, args):
    model_device = next(model.parameters()).device
    loss_fn = torch.nn.MSELoss()

    # test
    test_loss = 0
    test_sample_num = 0
    test_pred = []
    test_gt = []
    model.eval()
    model.train_ = False
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            inputs, t, targets = batch
            inputs = inputs.to(model_device)
            targets = targets.to(model_device)
            t = t.to(model_device)
            batch_size = inputs.shape[0]
            outputs = model(inputs, t)
            loss = loss_fn(outputs.squeeze(), targets.squeeze())
            test_pred.extend(outputs.cpu().detach().numpy().flatten().tolist())
            test_gt.extend(targets.cpu().detach().numpy().flatten().tolist())
            test_loss += loss.item() * batch_size
            test_sample_num += batch_size
    # print(test_gt, test_pred)

    test_results = pd.DataFrame({
        'Predictions': test_pred,
        'Ground Truth': test_gt
    })
    excel_file_path = os.path.join(args.log_dir, f'{args.expname}_results.xlsx')
    if not os.path.exists(os.path.dirname(excel_file_path)):
        os.makedirs(os.path.dirname(excel_file_path))
    save_results_to_excel(test_results, excel_file_path, 'Test Set', mode='a')

    test_loss /= test_sample_num
    test_r2 = r2_score(test_gt, test_pred)
    test_accuracy = calculate_accuracy(test_pred, test_gt, args.threshold)
    model.train()
    return test_loss, test_r2, test_gt, test_pred, test_accuracy


def set_all_seed(seed=1):
    """Set seed for reproducibility."""
    # PyTorch random seed
    torch.manual_seed(seed)
    # CUDA randomness
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Additional configurations to enforce deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # NumPy random seed
    np.random.seed(seed)
    # Python random seed
    random.seed(seed)
    
    
if __name__ == '__main__':
    args = parse_args()
    set_all_seed(args.seed)
    check_dir(os.path.join(args.log_dir, args.expname))
    check_dir(os.path.join(args.ckp_dir, args.expname))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
=======
import torch.optim as optim
import tqdm
import numpy as np
import torch
import os
from modules.dyformer import build_model
from data_processing.dataset import build_dataset
from parsing import parse_args
from sklearn.metrics import r2_score
import copy
import datetime
from data_processing.utils import check_dir
import random
import math
import time
import pandas as pd


def write_log_head(log_file,  args):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S\n")
    log_file.write(formatted_time)
    log_file.write('\n')
    log_file.write('Config\n')
    config_line = ''
    #* Log the arguments
    log_file.write('Experiment Configuration:')
    for arg, value in vars(args).items():
        log_file.write(f'{arg}: {value}')
    
    log_file.write(f'{config_line}\n')
    log_file.write('\n')
    
    
def write_log(log_file, performance):
    line = ''
    for key, value in performance.items():
        line += f'{key} - {value} '
    log_file.write(f'{line}\n')
    log_file.write('\n')
    return line


def calculate_accuracy(preds, labels, thresh=4 / 12.0):
    preds = np.array(preds)
    labels = np.array(labels)
    #! Check
    preds_class = np.where(preds >= thresh, 1, 0)
    labels_class = np.where(labels >= thresh, 1, 0)
    correct_count = np.sum(preds_class == labels_class)
    accuracy = correct_count / len(labels_class)
    return accuracy * 100


def save_results_to_excel(data, file_name, sheet_name, mode='w'):
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with pd.ExcelWriter(file_name, mode=mode, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name=sheet_name)


def train(args):
    # experiment log file
    log_file = open(os.path.join(f'{args.log_dir}/{args.expname}', f'log.txt'), 'w')
    write_log_head(log_file, args)
    # build dataset
    train_set, val_set, test_set = build_dataset(dataset_dir=args.dataset_dir,
                                                 series=args.series, batch_size=args.batch_size,
                                                 startpoint=args.startpoint, endpoint=args.endpoint, steps=args.steps,
                                                 resize_shape=args.resize_shape,
                                                 shuffle=True, split=args.split)

    # build model
    model = build_model(args)

    model_device = next(model.parameters()).device
    print(f'Training on {model_device}')
    print('=' * 20)
    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=args.step_size,
                                                gamma=args.gamma)
    loss_fn = torch.nn.MSELoss()
    start_epoch = 1

    best_model = None
    best_r2 = -99999.0
    best_data = {'train_gt': [], 'train_pred': [],
                 'val_gt': [], 'val_pred': [],
                 'test_gt': [], 'test_pred': []}
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # train
        train_loss = 0
        train_pred = []
        train_gt = []
        train_sample_num = 0
        model.train()
        model.train_ = True
        for batch in tqdm.tqdm(train_set):
            inputs, t, targets = batch
            inputs = inputs.to(model_device)
            targets = targets.to(model_device)
            t = t.to(model_device)
            batch_size = inputs.shape[0]
            outputs = model(inputs, t)
            optimizer.zero_grad()
            loss = loss_fn(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_pred.extend(outputs.cpu().detach().numpy().flatten())
            train_gt.extend(targets.cpu().detach().numpy().flatten())
            train_loss += loss.item() * batch_size
            train_sample_num += batch_size
        train_loss /= train_sample_num
        train_r2 = r2_score(y_true=train_gt, y_pred=train_pred)
        train_accuracy = calculate_accuracy(train_pred, train_gt, args.threshold)

        # validation
        val_loss = 0
        val_sample_num = 0
        val_pred = []
        val_gt = []
        model.eval()
        model.train_ = False
        with torch.no_grad():
            for batch in tqdm.tqdm(val_set):
                inputs, t, targets = batch
                inputs = inputs.to(model_device)
                targets = targets.to(model_device)
                t = t.to(model_device)
                batch_size = inputs.shape[0]
                outputs = model(inputs, t)
                loss = loss_fn(outputs.squeeze(), targets.squeeze())

                val_pred.extend(outputs.cpu().detach().numpy().flatten().tolist())
                val_gt.extend(targets.cpu().detach().numpy().flatten().tolist())
                val_loss += loss.item() * batch_size
                val_sample_num += batch_size

        # print(val_pred, val_gt)
        val_loss /= val_sample_num
        val_r2 = r2_score(y_true=val_gt, y_pred=val_pred)
        val_accuracy = calculate_accuracy(val_pred, val_gt, args.threshold)

        performance = {'Epoch': epoch, 'Train RMSE': math.sqrt(train_loss), 'Train R2': train_r2,
                       'Val RMSE': math.sqrt(val_loss), 'Val R2': val_r2, 'Train Acc': train_accuracy,
                       'Val Acc': val_accuracy}
        line = write_log(log_file, performance)
        print(line)

        if val_r2 > best_r2:
            best_model = copy.deepcopy(model)
            best_r2 = val_r2
            best_data['train_gt'] = train_gt
            best_data['train_pred'] = train_pred
            best_data['val_gt'] = val_gt
            best_data['val_pred'] = val_pred

            excel_file_path = os.path.join(args.log_dir, f'{args.expname}_results.xlsx')
            train_results = pd.DataFrame({
                'Train Predictions': best_data['train_pred'],
                'Train Ground Truth': best_data['train_gt']
            })
            val_results = pd.DataFrame({
                'Validation Predictions': best_data['val_pred'],
                'Validation Ground Truth': best_data['val_gt']
            })
            save_results_to_excel(train_results, excel_file_path, 'Train Set')
            save_results_to_excel(val_results, excel_file_path, 'Validation Set', mode='a')

        print('=' * 20)
    test_loss, test_r2, test_gt, test_pred, test_accuracy = test(best_model, test_set, args)
    performance['Test RMSE'] = math.sqrt(test_loss)
    performance['Test R2'] = test_r2
    performance['Test Acc'] = test_accuracy
    log_file.write('Test set performance\n')
    line = write_log(log_file, performance)
    print('Test')
    print(line)

    best_data['test_gt'] = test_gt
    best_data['test_pred'] = test_pred
    checkpoint_data = {
        'model': best_model.state_dict(),
        'val_r2': best_r2,
        'test_r2': test_r2,
        'results': best_data,
    }
    torch.save(checkpoint_data, os.path.join(args.ckp_dir, f'{args.expname}/best_model.pth'))

    #time record
    end_time = time.time()
    total_time = end_time - start_time
    log_file.write(f'Total training time: {total_time} seconds\n')

    # GPU consumed
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_usage = torch.cuda.memory_allocated() / gpu_memory
        log_file.write(f'GPU memory usage: {gpu_usage * 100:.2f}%\n')

    log_file.close()

    return checkpoint_data


def test(model, dataset, args):
    model_device = next(model.parameters()).device
    loss_fn = torch.nn.MSELoss()

    # test
    test_loss = 0
    test_sample_num = 0
    test_pred = []
    test_gt = []
    model.eval()
    model.train_ = False
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            inputs, t, targets = batch
            inputs = inputs.to(model_device)
            targets = targets.to(model_device)
            t = t.to(model_device)
            batch_size = inputs.shape[0]
            outputs = model(inputs, t)
            loss = loss_fn(outputs.squeeze(), targets.squeeze())
            test_pred.extend(outputs.cpu().detach().numpy().flatten().tolist())
            test_gt.extend(targets.cpu().detach().numpy().flatten().tolist())
            test_loss += loss.item() * batch_size
            test_sample_num += batch_size
    # print(test_gt, test_pred)

    test_results = pd.DataFrame({
        'Predictions': test_pred,
        'Ground Truth': test_gt
    })
    excel_file_path = os.path.join(args.log_dir, f'{args.expname}_results.xlsx')
    if not os.path.exists(os.path.dirname(excel_file_path)):
        os.makedirs(os.path.dirname(excel_file_path))
    save_results_to_excel(test_results, excel_file_path, 'Test Set', mode='a')

    test_loss /= test_sample_num
    test_r2 = r2_score(test_gt, test_pred)
    test_accuracy = calculate_accuracy(test_pred, test_gt, args.threshold)
    model.train()
    return test_loss, test_r2, test_gt, test_pred, test_accuracy


def set_all_seed(seed=1):
    """Set seed for reproducibility."""
    # PyTorch random seed
    torch.manual_seed(seed)
    # CUDA randomness
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Additional configurations to enforce deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # NumPy random seed
    np.random.seed(seed)
    # Python random seed
    random.seed(seed)
    
    
if __name__ == '__main__':
    args = parse_args()
    set_all_seed(args.seed)
    check_dir(os.path.join(args.log_dir, args.expname))
    check_dir(os.path.join(args.ckp_dir, args.expname))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>>>>>>> 12dbf5c (revision)
    train(args)