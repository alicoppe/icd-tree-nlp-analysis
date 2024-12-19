import torch
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.decomposition import PCA
from tqdm import tqdm 

class BERTWrapper(torch.nn.Module):
    def __init__(self, model_type="bert-base-uncased", random_init=False, device="cpu"):
        """
        Initialize the BERT model wrapper.
        
        Args:
        - model_type (str): Pretrained model name or path (e.g., 'bert-base-uncased' or 'emilyalsentzer/Bio_ClinicalBERT').
        - random_init (bool): If True, initializes BERT with random weights.
        - device (str): Device to load the model ('cpu' or 'cuda').
        """
        super(BERTWrapper, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_type)

        if random_init:
            config = BertConfig.from_pretrained(model_type)
            self.model = BertModel(config).to(self.device)
        else:
            self.model = BertModel.from_pretrained(model_type).to(self.device)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the BERT model.
        
        Args:
        - input_ids (torch.Tensor): Token IDs.
        - attention_mask (torch.Tensor): Attention mask.
        
        Returns:
        - torch.Tensor: Last hidden state of the BERT model.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def embed_text(self, text, reduce_dim=None, reduce_method="pca"):
        """
        Generate a full-text embedding for the input string.
        
        Args:
        - text (str): Input string to embed.
        - reduce_dim (int, optional): Dimensionality of the reduced embedding. If None, no reduction is applied.
        - reduce_method (str): Method for dimensionality reduction ('pca').
        
        Returns:
        - torch.Tensor: Text embedding (optionally reduced in dimensionality).
        """
        # Tokenize and encode the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.forward(inputs["input_ids"], inputs["attention_mask"])
        
        # Compute the mean embedding over the sequence
        full_embedding = embeddings.mean(dim=1).squeeze(0)

        # Perform dimensionality reduction if requested
        if reduce_dim is not None and reduce_method.lower() == "pca":
            full_embedding_np = full_embedding.cpu().numpy()
            pca = PCA(n_components=reduce_dim)
            reduced_embedding = pca.fit_transform(full_embedding_np.reshape(1, -1))
            return torch.tensor(reduced_embedding, dtype=torch.float32)
        
        return full_embedding

    def embed_text_array(self, text_array, reduce_dim=None, reduce_method="pca"):
        """
        Generate embeddings for an array of input strings.
        
        Args:
        - text_array (list of str): List of input strings to embed.
        - reduce_dim (int, optional): Dimensionality of the reduced embeddings. If None, no reduction is applied.
        - reduce_method (str): Method for dimensionality reduction ('pca').
        
        Returns:
        - list of torch.Tensor: List of embeddings for each input string.
        """
        embeddings = []
        for text in tqdm(text_array, desc="Embedding Text"):
            embedding = self.embed_text(text, reduce_dim, reduce_method)
            embeddings.append(embedding)
        
        return embeddings