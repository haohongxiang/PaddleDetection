import torch
import paddle

import numpy as np


@paddle.no_grad()
def clip_grad_norm_(params, max_norm, norm_type=2, error_if_nonfinite=True):
    '''clip_grad_norm_
    '''
    if isinstance(params, paddle.Tensor):
        params = [params]
    params = [p for p in params if p.stop_gradient is False]
    
    if len(params) == 0:
        return 0.
    
    max_norm = float(max_norm)
    
    total_norm = paddle.norm(paddle.stack([paddle.norm(x.grad, norm_type) for x in params]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in params:
            p.grad.set_value( p.grad * clip_coef )
    
    if total_norm.isnan() or total_norm.isinf():
        if error_if_nonfinite:
            raise RuntimeError('')
        else:
            print('Non-finite norm encountered')
            
    return total_norm


def get_torch_mm():
    m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, 2, 1),
            # torch.nn.BatchNorm2d(8),
            torch.nn.Flatten(),
            torch.nn.Linear(10 * 10 * 8, 10),
            # torch.nn.ReLU()
        )
    # m[0].weight.requires_grad = False
    
    return m


def get_paddle_mm():
    
    weight_attr = paddle.ParamAttr(learning_rate=0.1)
    bias_attr = paddle.ParamAttr(learning_rate=0.1)

    m = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, 8, 3, 2, 1, weight_attr=weight_attr, bias_attr=bias_attr),
            # paddle.nn.BatchNorm2D(8),
            paddle.nn.Flatten(),
            paddle.nn.Linear(10 * 10 * 8, 10),
            # paddle.nn.ReLU()
        )
    # m[0].weight.stop_gradient = True
    
    return m


def reset_parameters(paddle_model, torch_model, data=None):
    '''reset parameters
    '''
    sublayers = [(n, m) for n, m in paddle_model.named_sublayers(include_self=True)]
    submoduls = [(n, m) for n, m in torch_model.named_modules()]
    
    print(len(sublayers), len(submoduls))
    assert len(sublayers) == len(submoduls), f'{len(sublayers)}, {len(submoduls)}'
    
    for i, (n, m) in enumerate(sublayers):

        _m = submoduls[i][1]
        
        if isinstance(m, paddle.nn.Conv2D):
            m.weight.set_value( _m.weight.data.numpy() )
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value( _m.bias.data.numpy() )
                
        elif isinstance(m, paddle.nn.Linear):
            m.weight.set_value( _m.weight.data.numpy().T )
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value( _m.bias.data.numpy() )
        
        elif isinstance(m, paddle.nn.BatchNorm2D):
            m.weight.set_value( _m.weight.data.numpy() )
            if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
                m.bias.set_value( _m.bias.data.numpy() )
            
            m._mean.set_value( _m.running_mean.data.numpy() )
            m._variance.set_value( _m.running_var.data.numpy() )
    
    
    if data is None:
        data = np.random.rand(1, 3, 20, 20).astype(np.float32)

    tdata = torch.tensor(data[...])
    pdata = paddle.to_tensor(data[...])
    tdata.requires_grad = True
    pdata.stop_gradient = False

    tout = torch_model(tdata)
    pout = paddle_model(pdata)

    print('-------forward---------')
    print('torch', tout.mean(), tout.sum())
    print('paddle', pout.mean(), pout.sum())
    np.testing.assert_almost_equal(pout.numpy(), tout.data.numpy(), decimal=4)

    print('-------backward---------')
    tout.sum().backward()
    pout.sum().backward()

    print('torch', tdata.grad.mean(), tdata.grad.sum())
    print('paddle', pdata.grad.mean(), pdata.grad.sum())
    np.testing.assert_almost_equal(pdata.grad.numpy(), tdata.grad.data.numpy(), decimal=4)

    check_parameters(paddle_model, torch_model)
    print('-------<after checking forward, backward for input and parameters>----------')
    print('----------------------------reset parameters done---------------------------')

    
    
def check_parameters(paddle_model, torch_model, decimal=4):
    '''check_parameters
    '''
    tnp = [(n, p) for n, p in torch_model.named_parameters() if p.requires_grad]
    pnp = [(n, p) for n, p in paddle_model.named_parameters() if not p.stop_gradient]
    assert len(tnp) == len(pnp), ''
    
    for _tnp, _pnp in zip(tnp, pnp):
        _tp = _tnp[-1]; _pp = _pnp[-1]
        # print(_tnp[0], _pnp[0])
        
        if list(_tp.shape) == list(_pp.shape):
            np.testing.assert_almost_equal(_tp.data.numpy(), _pp.numpy(), decimal=decimal)
            np.testing.assert_almost_equal(_tp.grad.data.numpy(), _pp.grad.numpy(), decimal=decimal)
        elif list(_tp.shape[::-1]) == list(_pp.shape):
            np.testing.assert_almost_equal(_tp.data.numpy().T, _pp.numpy(), decimal=decimal)
            np.testing.assert_almost_equal(_tp.grad.data.numpy().T, _pp.grad.numpy(), decimal=decimal)
        else:
            raise RuntimeError('--')

            

def check_optimizer(paddle_model, torch_model, optim_name):
    '''check_optimizer
    '''
    
    epoches = 5
    iters_per_epoch = 3
    
    lr = 0.1
    gamma = 0.1
    milestones = [2, 4]
    
    
    if True:
        tp = [p for p in torch_model.parameters() if p.requires_grad]
        pp = [p for p in paddle_model.parameters() if not p.stop_gradient]
        assert len(tp) == len(pp), ''
        
        tp = [{'params': torch_model[0].parameters(), 'lr': lr * 0.1}, {'params': torch_model[-1].parameters(),}]

    else:
        tp = torch_model.parameters()
        pp = paddle_model.parameters()
        
        
    pscheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr, milestones=milestones, gamma=gamma)

    # pclip = paddle.nn.ClipGradByGlobalNorm(0.1)
    paddle_optimizers = {
        'AdamW': paddle.optimizer.AdamW(learning_rate=pscheduler, parameters=pp, weight_decay=0.01, ), # grad_clip=pclip),
        'SGD': paddle.optimizer.SGD(learning_rate=pscheduler, parameters=pp, weight_decay=0.0),
    }
    
    torch_optimizers = {
        'AdamW': torch.optim.AdamW(tp, lr=lr, weight_decay=0.01),
        'SGD': torch.optim.SGD(tp, lr=lr, weight_decay=0.0),
    }

    toptim = torch_optimizers[optim_name]; 
    poptim = paddle_optimizers[optim_name]

    tscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=toptim, milestones=milestones, gamma=gamma)
    
    for e in range(epoches):
        for i in range(iters_per_epoch):
            data = np.random.rand(1, 3, 20, 20).astype(np.float32) * 2 - 1
            
            tdata = torch.tensor(data[...])
            pdata = paddle.to_tensor(data[...])
    
            tout = torch_model(tdata)
            pout = paddle_model(pdata)    
            
            toptim.zero_grad()
            poptim.clear_grad()
            
            tout.sum().backward()
            pout.sum().backward()
            
            tnorm = torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 0.1)
            pnorm = clip_grad_norm_(paddle_model.parameters(), 0.1)
            print(e, i, tnorm, pnorm)

            toptim.step()
            poptim.step()
            
            print('{}/{}'.format(e, i), tout.sum().data.numpy(), pout.sum().numpy())
            print()
            check_parameters(paddle_model, torch_model, decimal=2)
            
        tscheduler.step()
        pscheduler.step()
        
        print(e, toptim.param_groups[0]['lr'], pscheduler.get_lr(),)
        
            
if __name__ == '__main__':
    
    tm = get_torch_mm()
    pm = get_paddle_mm()

    reset_parameters(pm, tm)

    check_optimizer(pm, tm, 'AdamW')
    # check_optimizer(pm, tm, 'SGD')


