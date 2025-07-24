import torch
from train_test import Tester_BAN
from src.utils import set_random_seed
from src.utils import get_featurizer
from src.data import get_task_dir, DTADataModule
from src.model import proSeqEncoder
from src.gvp_gnn import StructureEncoder
from src import model as model_types

# Load best model
best_model_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# Set random seed
set_random_seed(42, deterministic=True)

# Load dummy config values
drug_dim = 256
target_dim = 1280
h_dim = 128
n_heads = 6

# Define dummy shapes
drug_seq_shape = [300]
drug_struc_shape = [256]
target_seq_shape = [1024]
node_in_dim = (6, 3)
node_h_dim = (128, 3)
edge_in_dim = (3, 2)
edge_h_dim = (32, 2)

# Encoders
drug_seq_encoder = proSeqEncoder(300, drug_dim, 1, 5, 61, dropout=0.3)
target_struc_encoder = StructureEncoder(
    node_in_dim=node_in_dim,
    node_h_dim=node_h_dim,
    edge_in_dim=edge_in_dim,
    edge_h_dim=edge_h_dim,
    seq_in=False,
    num_layers=3,
    drop_rate=0.3,
)

# Model
model = model_types.PocketBAN(
    drug_seq_encoder,
    target_seq_shape,
    target_struc_encoder,
    drug_seq_dim=drug_seq_shape,
    drug_dim=drug_dim,
    target_dim=target_dim,
    gvp_output_dim=node_h_dim[0],
    h_dim=h_dim,
    n_heads=n_heads,
    use_drug_seq=True,
    use_drug_struc=True,
    use_target_seq=True,
    use_target_struc=True,
).to(device)

# Load model weights
model.load_state_dict(torch.load(best_model_path))
tester = Tester_BAN(model, batch_size)

# Dummy test dataloader
task_dir = get_task_dir("Davis")
drug_seq_featurizer = get_featurizer("mol2vec", save_dir=task_dir)
drug_struc_featurizer = get_featurizer("GraphMVP", name="graphmvp", save_dir=task_dir)
target_seq_featurizer = get_featurizer("protbert", save_dir=task_dir)

datamodule = DTADataModule(
    task_dir,
    drug_seq_featurizer,
    drug_struc_featurizer,
    target_seq_featurizer,
    tops="path/to/pocket.json",
    device=device,
    seed=42,
    use_cold_spilt=False,
    use_test=True,
    cold=["Drug", "target_key"],
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

datamodule.prepare_data()
datamodule.setup()

# Run test
testing_generator, test_len = datamodule.test_dataloader(domain=True)
G_test, P_test, att = tester.test(testing_generator, device, test_len)

# Attention processing
def process_tensor(tensor):
    heads_mean = tensor.mean(dim=1).squeeze(0)
    drug_mean = heads_mean.mean(dim=1).squeeze(0)
    target_mean = heads_mean.mean(dim=0).squeeze(0)
    normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    return normalize(drug_mean), normalize(target_mean)

drug_softmax, target_softmax = process_tensor(att)
print("Drug attention:", drug_softmax)
print("Target attention:", target_softmax)
