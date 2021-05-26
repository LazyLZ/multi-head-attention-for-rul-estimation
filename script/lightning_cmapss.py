import torch
import pytorch_lightning as pl
from dataset import CMAPSSDataset
from model import MultiHeadAttentionLSTM
from torch.nn import functional as F
import pandas as pd
from pytorch_lightning.metrics.functional import mean_squared_error
from utils.metrics import score


class Module(pl.LightningModule):
    def __init__(self, lr, save_weights, **kwargs):
        super(Module, self).__init__()
        self.net = MultiHeadAttentionLSTM(**kwargs)
        self.lr = lr
        self.save_weights = save_weights
        # print(self.net)
        # print('lr', self.lr)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        # print('x', x)
        # print('y', y)
        x, _ = self.net(x)
        loss = F.mse_loss(x, y)
        self.log('train_rmse', torch.sqrt(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x, _ = self.net(x)
        loss = F.mse_loss(x, y, reduction='sum')
        return torch.tensor([loss.item(), len(y)])

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y, id = batch
        x, feature_weight = self.net(x)
        if feature_weight is not None:
            return torch.cat([id, x, y, feature_weight.view(feature_weight.size(0), -1)], dim=1)
        else:
            return torch.cat([id, x, y], dim=1)

    def test_epoch_end(self, step_outputs):
        t = torch.cat(step_outputs, dim=0)
        t = t.cpu()
        if self.save_weights and t.shape[1] > 3:
            f_attn_ij = []
            feature_num = self.net.feature_num
            for i in range(feature_num):
                for j in range(feature_num):
                    f_attn_ij.append('%d_%d' % (i, j))
            data = t[:, : 3 + len(f_attn_ij)]
            df = pd.DataFrame(data.numpy(), columns=['id', 'output', 'label'] + f_attn_ij)
            print()
            print(df)
            df.to_csv('test_result_with_feature_attention_weight.csv', index=False)

        if self.save_weights and t.shape[1] <= 3:
            print('\n[SaveAttentionWeights] Attention is not used')
        # print('test size', t.shape[0])
        rmse = torch.sqrt(mean_squared_error(t[:, 1], t[:, 2]))
        s = score(t[:, 1], t[:, 2])
        self.log('test_rmse', rmse)
        self.log('test_score', s)

    def validation_epoch_end(self, val_step_outputs):
        t = torch.stack(val_step_outputs)
        t = torch.sum(t, dim=0)
        self.log('val_rmse', torch.sqrt(t[0] / t[1]), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=24)
    parser.add_argument('--hidden-dim', type=int, default=100, help='RNN hidden dims')
    parser.add_argument('--cell', type=str, default='lstm', help='lstm, gru or rnn')
    parser.add_argument('--fc-layer-dim', type=int, default=100)
    parser.add_argument('--rnn-num-layers', type=int, default=3)
    parser.add_argument('--fc-activation', type=str, default='relu', help='relu, tanh or gelu')
    parser.add_argument('--attention-order', action='append', help='value must be "feature"')
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--feature-head-num', type=int, default=10)
    parser.add_argument('--fc-dropout', type=float, default=0.5)
    parser.add_argument('--save-attention-weights', action='store_true', default=False)
    parser.add_argument('--dataset-root', type=str, required=True, help='The dir of CMAPSS dataset')
    parser.add_argument('--sub-dataset', type=str, required=True, help='FD001/2/3/4')
    parser.add_argument('--norm-type', type=str, help='z-score, -1-1 or 0-1')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--cluster-operations', action='store_true', default=False)
    parser.add_argument('--norm-by-operations', action='store_true', default=False)
    parser.add_argument('--use-max-rul-on-test', action='store_true', default=True)
    parser.add_argument('--validation-rate', type=float, default=0, help='validation set ratio of train set')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args = parser.parse_args()
    model_kwargs = {
        'sequence_len': args.sequence_len,
        'feature_num': args.feature_num,
        'hidden_dim': args.hidden_dim,
        'cell': args.cell,
        'fc_layer_dim': args.fc_layer_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'output_dim': 1,
        'fc_activation': args.fc_activation,
        'attention_order': args.attention_order or [],
        'bidirectional': args.bidirectional,
        'feature_head_num': args.feature_head_num,
        'fc_dropout': args.fc_dropout,
        'return_attention_weights': True
    }
    train_loader, test_loader, valid_loader = CMAPSSDataset.get_data_loaders(
        dataset_root=args.dataset_root,
        sequence_len=args.sequence_len,
        sub_dataset=args.sub_dataset,
        norm_type=args.norm_type,
        max_rul=args.max_rul,
        cluster_operations=args.cluster_operations,
        norm_by_operations=args.norm_by_operations,
        use_max_rul_on_test=args.use_max_rul_on_test,
        validation_rate=args.validation_rate,
        return_id=True,
        use_only_final_on_test=not args.save_attention_weights,
        loader_kwargs={'batch_size': args.batch_size}
    )

    model = Module(
        lr=args.lr,
        save_weights=args.save_attention_weights,
        **model_kwargs
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_rmse',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_rmse',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        default_root_dir='../checkpoints',
        gpus=1 if not args.no_cuda else None,
        max_epochs=args.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        # checkpoint_callback=False,
        # logger=False,
        # progress_bar_refresh_rate=0
    )
    trainer.fit(model, train_loader, val_dataloaders=valid_loader or test_loader)
    trainer.test(test_dataloaders=test_loader)
