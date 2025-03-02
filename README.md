# Name: Swaraj Bhanja | Student Id: st125052

# Welcome to Direct Preference Optimization!

This is a web-based end-to-end application named Direct Preference Optimization. It leverages the power of deep learning and web development to provide a website that performs text generation based on the input sentence or phrase.

# About the Deep Learning Model

The brains of this solution is the deep learning model trained for this purpose. The DL model was trained based on the HuggingFace **GPT2** pre-trained model being subject to Reinforcement Human Learning Feedback. The **GPT2** pre-trained model is a large-scale Transformer-based language model trained on vast internet text data. This model is capable of generating coherent and contextually relevant text by predicting the next token in a sequence. The dataset was originally created to train large-scale language models, such as those used in text generation, language understanding, and transfer learning tasks. 

# Loading the Pre-Trained Model & Its Tokenizer

A pre-trained language model and tokenizer from the Hugging Face Transformers library is used here. The model used is `GPT-2`, specified by the variable model_name_or_path. First, an instance of `AutoModelForCausalLM` is created using `from_pretrained`, which loads the model weights from Hugging Face's pre-trained repository. A conditional check follows, where if `ignore_bias_buffers` is set to True, the code attempts to exclude certain boolean-type buffer parameters from being tracked by PyTorch’s distributed data parallel (DDP) mechanism, which is often useful in multi-GPU training scenarios. Another identical instance of the `model`, `model_ref`, is also loaded to serve as a reference for comparison during training or evaluation. The tokenizer corresponding to the model is then instantiated using AutoTokenizer.from_pretrained. If the tokenizer lacks a designated padding token (often required for batch processing), it assigns the end-of-sequence `(eos_token)` as the padding token to prevent errors in tokenized input handling.

# Loading the RHLF Dataset

The Anthropic Helpful-Harmless (HH-RLHF) dataset is then loaded and processed, which is used for training AI models to generate safe and helpful responses. The function `extract_anthropic_prompt` is designed to extract only the prompt portion from a combined prompt-and-response text. It does this by searching for the separator `"\n\nAssistant:"`, which indicates where the assistant’s response begins. If this separator is not found, the function raises an error, ensuring that the dataset format is correct.

The get_hh function loads the dataset from Hugging Face `(Anthropic/hh-rlhf)`, with an optional sanity check to limit the dataset size to 1,000 samples for quick validation. The dataset consists of human prompts and two responses: one preferred `("chosen")` and one rejected `("rejected")`. Inside `get_hh`, the function `split_prompt_and_responses` ensures that the prompt is separated from the response by identifying where the assistant’s reply begins. It then structures the dataset into a dictionary format with three keys: `"prompt"` (the input question or statement), `"chosen"` (the assistant’s preferred response), and `"rejected"` (the response deemed less helpful or safe). Finally, the function maps this transformation onto the dataset. The script loads both the training `(train_dataset)` and evaluation `(eval_dataset)` sets, ensuring that they are preprocessed and structured correctly before being used in further experiments.

# Initializing Training Argument

A set of training configurations for fine-tuning a model using Direct Preference Optimization (DPO) is initialized. The configuration is created using DPOConfig, which comes from a deep learning framework i.e. the Hugging Face's TRL library.

The training process is set for 3 epochs. The learning rate is set to `5e-7`, which is very small to ensure stable training without large weight updates. The `batch size per device` is 1, meaning only one example is processed per forward pass, both for training and evaluation. The `Adam optimizer’s` epsilon value `(1e-8)` is set to maintain numerical stability in gradient updates.

A `linear learning rate` scheduler is used, meaning the learning rate will gradually decrease over time. The `warmup ratio` of 0.1 ensures that for the first 10% of training steps, the learning rate will slowly ramp up before stabilizing, preventing large gradient updates early on. The random seed is fixed at 42 for reproducibility. Training progress is logged every 100 steps, and model checkpoints are saved every 100 steps using a "steps"-based saving strategy.

The model checkpoints and logs will be stored in the `./output-dir` directory. Gradient checkpointing is enabled, which helps reduce memory consumption by recomputing activations during backpropagation instead of storing them. The `bf16=True` flag enables bfloat16 precision, which speeds up training while maintaining numerical accuracy. Finally, `remove_unused_columns=False` ensures that all dataset columns are preserved during training, preventing automatic column removal by the trainer.

# Training

A DPO trainer instance is initialized, which is responsible for the RHLF process. The DPOTrainer is configured with several key components: the main model `(model)`, a reference model `(model_ref)`, the training arguments `(training_args)`, and the datasets `(train_dataset and eval_dataset)`. The reference model acts as a baseline for comparison, meaning the fine-tuned model will be optimized to prefer certain responses over others based on human feedback.

The `train_dataset` contains training examples, while the `eval_dataset` is used for evaluating the model’s performance at different checkpoints. The `processing_class=tokenizer` ensures that input text is properly tokenized before being fed into the model. The DPOTrainer handles the training loop, performing backpropagation, updating model weights, logging metrics, and saving model checkpoints according to the settings defined in `training_args`.

## Testing

An input_text variable contains the phrase `"I am an AI engineer"`, which serves as the initial prompt for the model. The tokenizer processes this text using `tokenize(input_text, return_tensors="pt")`, converting it into a PyTorch tensor ("pt" stands for PyTorch), which is the format required for deep learning models.

The `inputs.to(device)` command ensures that the tokenized input is moved to the correct device (CPU or GPU) for processing. The `torch.no_grad()` context manager disables gradient calculations to save memory and speed up inference, since we are not training the model, just generating text.

The `model.generate(**inputs, max_length=100)` function generates text using the causal language model, continuing from the given prompt and producing up to 100 tokens. The output is a tensor of token IDs, which needs to be converted back into readable text using `tokenizer.decode(output[0], skip_special_tokens=True)`. This removes any special tokens `(like <|endoftext|>)` that may appear in the generated output. Finally, the generated text is printed, showing the model’s continuation of the initial input.

[Analysis Metrics](https://github.com/st125052/a5-nlp-direct-preference-optimization-st125052/blob/main/notebooks/pdfs/Training%20Metrics%20Based%20on%20Hyperparameter%20Experimentation.pdf)

## Uploading The Model To Hugging Face

The model was then uploaded to HuggingFace to run inference via the HuggingFace Transformers library in Python. The visibility of the model was kept public for open-source access.

[Check out the model](https://huggingface.co/st125052/a5-dpo)

# Website Creation
The model was then hosted over the Internet with Flask as the backend, HTML, CSS, JS as the front end, and Docker as the container. The end-user is presented with a UI wherein a search input box is present. Once the user types in the first set of words, they click on the `Generate Response` button. The input texts are sent to the JS handler which makes an API call to the Flask backend. The Flask backend has the GET route which intercepts the HTTP request. The input text is then fed to the model to generate the response. The model and tokenizer are also cached in the RAM in this process for faster inference in the successive runs. The result is then returned back to the JS handler as a list by the Flask backend. The JS handler then appends each token in the received list into the result container's inner HTML and finally makes it visible for the output to be shown. 

A Vanilla architecture was chosen due to time constraints. In a more professional scenario, the ideal approach would be used frameworks like React, Angular and Vue for Frontend and ASP.NET with Flask or Django for Backend.

The following describes the key points of the hosting discussion.
> **1. DigitalOcean (Hosting Provider)**
> 
>> - **Role:** Hosting and Server Management
>> - **Droplet:** Hosts the website on a virtual server, where all files, databases, and applications reside.
>> - **Dockerized Container:** The website is hosted in a Dockerized container running on the droplet. The container is built over a Ubuntu Linux 24.10 image.
>> - **Ports and Flask App:** The Dockerized container is configured to host the website on port 8000. It forwards requests to port 5000, where the Flask app serves the backend and static files. This flask app contains the pickled model, which is used for prediction.
>> - **IP Address:** The droplet’s public IP address directs traffic to the server.
>
>  **In Summary:** DigitalOcean is responsible for hosting the website within a Dockerized container, ensuring it is online and accessible via its IP address.
> 
>  **2. GoDaddy (Domain Registrar)**
>
>> - **Role:** Domain Registration and Management
>> - **Domain Purchase:** Registers and manages the domain name.
>> - **DNS Management:** Initially provided DNS setup, allowing the domain to be pointed to the DigitalOcean droplet’s IP address.
> 
> **In Summary:** GoDaddy ensures the domain name is registered and correctly points to the website’s hosting server.
>
>  **3. Cloudflare (DNS and Security/Performance Optimization)**
>
>> - **Role:** DNS Management, Security, and Performance Optimization
>> - **DNS Management:** Resolves the domain to the correct IP address, directing traffic to the DigitalOcean droplet.
>> - **CDN and Security:** Caches website content globally, enhances performance, and provides security features like DDoS protection and SSL encryption.
> 
> **In Summary:** Cloudflare improves the website’s speed, security, and reliability.
>
> **How It Works Together:**
> 
>> - **Domain Resolution:** The domain is registered with GoDaddy, which points it to Cloudflare's DNS servers. Cloudflare resolves the domain to the DigitalOcean droplet's IP address.
>> - **Content Delivery:** Cloudflare may serve cached content or forward requests to DigitalOcean, which processes and serves the website content to users.
> 
> **Advantages of This Setup:**
>
>> - **Security:** Cloudflare provides DDoS protection, SSL/TLS encryption, and a web application firewall.
>> - **Performance:** Cloudflare’s CDN reduces load times by caching content globally, while DigitalOcean offers scalable hosting resources.
>> - **Reliability:** The combination of GoDaddy, Cloudflare, and DigitalOcean ensures the website is always accessible, with optimized DNS resolution and robust hosting.

# Demo
https://github.com/user-attachments/assets/115e21e7-bb06-4970-828d-1dc7dcc3e51b


# Access The Final Website
You can access the website [here](https://aitmltask.online). 

# Limitations
Note that the model predicts only a slighly meaningful response to a certain length, beyond which it generates gibberish with the same repitition. Also, it may generate unwanted content for some contents, which is a known limitation


# How to Run the Direct Preference Optimization Docker Container Locally
### Step 1: Clone the Repository
> - First, clone the repository to your local machine.
### Step 2: Install Docker
> - If you don't have Docker installed, you can download and install it from the [Docker](https://www.docker.com) website.
### Step 3: Build and Run the Docker Container
Once Docker is installed, navigate to the app folder in the project directory. Delete the docker-compose-deployment.yml file and run the following commands to build and run the Docker container:
> - `docker compose up -d`

### Important Notes
> - The above commands will serve the Docker container on port 5000 and forward the requests to the Flask application running on port 5000 in the containerized environment.
> - Ensure Ports Are Free: Make sure that port 5000 is not already in use on your machine before running the container.
> - Changing Flask's Port: If you wish to change the port Flask runs on (currently set to 5000), you must update the port in the app.py file. After making the change, remember to rebuild the Docker image in the next step. Execute the following command to stop the process: `docker compose down`. Then goto Docker Desktop and delete the container and image from docker. 
