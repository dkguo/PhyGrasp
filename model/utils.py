import matplotlib.pyplot as plt
import pickle
import argparse

def save_loss(train_losses_global, test_losses_global, train_losses_local_pos, test_losses_local_pos, train_losses_local_neg, test_losses_local_neg, model_id):
    losses = {
        'train_losses_global': train_losses_global,
        'train_losses_local_pos': train_losses_local_pos,
        'train_losses_local_neg': train_losses_local_neg,
        'test_losses_global': test_losses_global,
        'test_losses_local_pos': test_losses_local_pos,
        'test_losses_local_neg': test_losses_local_neg
    }
    with open('./checkpoints/losses_{}.pkl'.format(model_id), 'wb') as f:
        pickle.dump(losses, f)

def save_loss10(train_losses_global, test_losses_global, train_losses_local_pos, test_losses_local_pos, train_losses_local_neg, test_losses_local_neg, train_losses_local_pos1, test_losses_local_pos1, train_losses_local_neg1, test_losses_local_neg1, model_id):
    losses = {
        'train_losses_global': train_losses_global,
        'test_losses_global': test_losses_global,
        'train_losses_local_pos': train_losses_local_pos,
        'test_losses_local_pos': test_losses_local_pos,
        'train_losses_local_neg': train_losses_local_neg,
        'test_losses_local_neg': test_losses_local_neg,
        'train_losses_local_pos1': train_losses_local_pos1,
        'test_losses_local_pos1': test_losses_local_pos1,
        'train_losses_local_neg1': train_losses_local_neg1,
        'test_losses_local_neg1': test_losses_local_neg1,
    }
    with open('./checkpoints/losses_{}.pkl'.format(model_id), 'wb') as f:
        pickle.dump(losses, f)

def save_loss2(train_losses_global, test_losses_global, model_id):
    losses = {
        'train_losses_global': train_losses_global,
        'test_losses_global': test_losses_global,
    }
    with open('./checkpoints/losses_{}.pkl'.format(model_id), 'wb') as f:
        pickle.dump(losses, f)

def load_loss(model_id):
    with open('./checkpoints/losses_{}.pkl'.format(model_id), 'rb') as f:
        losses = pickle.load(f)
    train_losses_global = losses['train_losses_global']
    train_losses_local_pos = losses['train_losses_local_pos']
    train_losses_local_neg = losses['train_losses_local_neg']
    test_losses_global = losses['test_losses_global']
    test_losses_local_pos = losses['test_losses_local_pos']
    test_losses_local_neg = losses['test_losses_local_neg']
    return train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg

def load_loss10(model_id):
    with open('./checkpoints/losses_{}.pkl'.format(model_id), 'rb') as f:
        losses = pickle.load(f)
    train_losses_global = losses['train_losses_global']
    train_losses_local_pos = losses['train_losses_local_pos']
    train_losses_local_neg = losses['train_losses_local_neg']
    test_losses_global = losses['test_losses_global']
    test_losses_local_pos = losses['test_losses_local_pos']
    test_losses_local_neg = losses['test_losses_local_neg']
    train_losses_local_pos1 = losses['train_losses_local_pos1']
    train_losses_local_neg1 = losses['train_losses_local_neg1']
    test_losses_local_pos1 = losses['test_losses_local_pos1']
    test_losses_local_neg1 = losses['test_losses_local_neg1']
    return train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg, train_losses_local_pos1, train_losses_local_neg1, test_losses_local_pos1, test_losses_local_neg1

def load_loss2(model_id):
    with open('./checkpoints/losses_{}.pkl'.format(model_id), 'rb') as f:
        losses = pickle.load(f)
    train_losses_global = losses['train_losses_global']
    test_losses_global = losses['test_losses_global']
    return train_losses_global, test_losses_global

def plot(train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg, model_id):
    '''
    plot 6 figures:
    with sparse data, the loss is not smooth
    '''
    # step_size = 1
    # train_losses_global = train_losses_global[::step_size]
    # train_losses_local_pos = train_losses_local_pos[::step_size]
    # train_losses_local_neg = train_losses_local_neg[::step_size]
    # test_losses_global = test_losses_global[::step_size]
    # test_losses_local_pos = test_losses_local_pos[::step_size]
    # test_losses_local_neg = test_losses_local_neg[::step_size]
    
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(train_losses_global)
    plt.title('train_losses_global')
    plt.subplot(2, 3, 2)
    plt.plot(train_losses_local_pos)
    plt.title('train_losses_local_pos')
    plt.subplot(2, 3, 3)
    plt.plot(train_losses_local_neg)
    plt.title('train_losses_local_neg')
    plt.subplot(2, 3, 4)
    plt.plot(test_losses_global)
    plt.title('test_losses_global')
    plt.subplot(2, 3, 5)
    plt.plot(test_losses_local_pos)
    plt.title('test_losses_local_pos')
    plt.subplot(2, 3, 6)
    plt.plot(test_losses_local_neg)
    plt.title('test_losses_local_neg')
    plt.savefig('./checkpoints/loss_{}.png'.format(model_id))

def plot2(train_losses_global, test_losses_global, model_id):
    '''
    plot 2 figures:
    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_global)
    plt.title('train_losses_global')
    plt.subplot(1, 2, 2)
    plt.plot(test_losses_global)
    plt.title('test_losses_global')
    plt.savefig('./checkpoints/loss_{}.png'.format(model_id))

def plot10(train_losses_global, test_losses_global, train_losses_local_pos, test_losses_local_pos, train_losses_local_neg, test_losses_local_neg, train_losses_local_pos1, test_losses_local_pos1, train_losses_local_neg1, test_losses_local_neg1, model_id):
    '''
    plot 10 figures:


    '''
    plt.figure()
    plt.subplot(2, 5, 1)
    plt.plot(train_losses_global)
    plt.title('train_global')
    plt.subplot(2, 5, 2)
    plt.plot(train_losses_local_pos)
    plt.title('train_pos')
    plt.subplot(2, 5, 3)
    plt.plot(train_losses_local_neg)
    plt.title('train_neg')
    plt.subplot(2, 5, 4)
    plt.plot(train_losses_local_pos1)
    plt.title('train_pos1')
    plt.subplot(2, 5, 5)
    plt.plot(train_losses_local_neg1)
    plt.title('train_neg1')
    plt.subplot(2, 5, 6)
    plt.plot(test_losses_global)
    plt.title('test_global')
    plt.subplot(2, 5, 7)
    plt.plot(test_losses_local_pos)
    plt.title('test_pos')
    plt.subplot(2, 5, 8)
    plt.plot(test_losses_local_neg)
    plt.title('test_neg')
    plt.subplot(2, 5, 9)
    plt.plot(test_losses_local_pos1)
    plt.title('test_pos1')
    plt.subplot(2, 5, 10)
    plt.plot(test_losses_local_neg1)
    plt.title('test_neg1')
    
    plt.savefig('./checkpoints/loss_{}.png'.format(model_id))


def main(params):
    with open('./checkpoints/losses_{}.pkl'.format(params['model_id']), 'rb') as f:
        losses = pickle.load(f)
    if len(losses) == 6:
        train_losses_global = losses['train_losses_global']
        train_losses_local_pos = losses['train_losses_local_pos']
        train_losses_local_neg = losses['train_losses_local_neg']
        test_losses_global = losses['test_losses_global']
        test_losses_local_pos = losses['test_losses_local_pos']
        test_losses_local_neg = losses['test_losses_local_neg']
        plot(train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg, model_id=params['model_id'])
    elif len(losses) == 10:
        train_losses_global = losses['train_losses_global']
        train_losses_local_pos = losses['train_losses_local_pos']
        train_losses_local_neg = losses['train_losses_local_neg']
        test_losses_global = losses['test_losses_global']
        test_losses_local_pos = losses['test_losses_local_pos']
        test_losses_local_neg = losses['test_losses_local_neg']
        train_losses_local_pos1 = losses['train_losses_local_pos1']
        train_losses_local_neg1 = losses['train_losses_local_neg1']
        test_losses_local_pos1 = losses['test_losses_local_pos1']
        test_losses_local_neg1 = losses['test_losses_local_neg1']
        plot10(train_losses_global, test_losses_global, train_losses_local_pos, test_losses_local_pos, train_losses_local_neg, test_losses_local_neg, train_losses_local_pos1, test_losses_local_pos1, train_losses_local_neg1, test_losses_local_neg1, model_id=params['model_id'])

    # if params['global']:
    #     train_losses_global, test_losses_global = load_loss2(params['model_id'])
    #     plot2(train_losses_global, test_losses_global, model_id=params['model_id'])
    # else:
    #     train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg = load_loss(params['model_id'])
    #     plot(train_losses_global, train_losses_local_pos, train_losses_local_neg, test_losses_global, test_losses_local_pos, test_losses_local_neg, model_id=params['model_id'])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_id', type=str, default='1705635392.6037188')
    argparser.add_argument('--global', default=False, action='store_true')

    args = argparser.parse_args()
    params = vars(args)
    main(params)