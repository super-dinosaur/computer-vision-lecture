import matplotlib.pyplot as plt

def draw_loss_curve(train_losses, best_miou, best_epoch, num_epochs, architecture, path_save):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.text(0.95, 0.95, f'Best mIoU: {best_miou:.4f} at Epoch {best_epoch}', ha='right', va='top', transform=plt.gca().transAxes, color='red')
    plt.savefig(path_save)