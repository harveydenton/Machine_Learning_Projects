## Feed forward NN for predicting Insurance Claims
This project involved training a feed forward neural network (NN) to predict the possibility of policyholders filing a claim.

The dataset consisted of 58,592 entries and 42 variables. The response was a binary variable “is_claim”, representing whether a policyholder would file a claim in the next 6 months. The covariates included both numerical and factor variables such as engine type, age of policyholder, population density, and max power of the car.

I first encoded variables with non-numerical datatypes, then converted the data into a tensor dataset with an 80/20 train-test split.

After training the NN and evaluating on the test set, I observed high test accuracy (92–95%). However,upon further inspection I noticed the model was not as effective as expected.
I introduced code to calculate accuracy for predicting claims and no-claims separately. The results were 100% for no-claims and 0% for claims.

This occurred because only 3,748 out of 58,592 policyholders filed a claim (6%). The model failed to learn the minority class and instead predicted “no claim” for all entries, explaining the high overall accuracy but low usefulness.

To improve performance, I increased model complexity by adding more layers and neurons. I also adjusted the classification threshold and analyzed its effect on performance by plotting the trade-off curve between “claims” and “no claims” accuracy.

Using these methods, I achieved more balanced results, with both accuracies reaching around 60%. By tuning the threshold, I obtained configurations such as 70% claims accuracy and 50% no-claims accuracy.

Overall, the NN was not an ideal fit. However, this project provided valuable experience handling imbalanced data (6% claims) and highlighted the importance of recall, specificity, and precision over standard accuracy, as well as threshold tuning.

I also recognized that NNs are not always suitable for tabular data. A brief attempt with XGBoost produced similar performance.




## CNN for estimating/solving sudoku puzzles

This project involved training a Convolutional neural network (CNN) to solve sudoku puzzles. This involved using a dataset of 4,000,000 sudoku puzzles with varying difficulty. 

I initially started with a dataset of sudoku puzzles of similar difficulty. This difficulty was relatively hard as the vast majority of cells in the puzzle were empty. I had also initially used a GNN as I preferred to have the rules and restraints of sudoku incorporated into the training of the Neural network.

This initial attempt however resulted in long training times with poor accuracy such as a cell accuracy of <25% and a puzzle accuracy of 0% (cells refer to the empty entries in each puzzle). I decided that the reason for this was the complexity of GNNs along with the model struggling to learn from the hard sudoku puzzles.

Therefore, for this new dataset, I decided to use curricular learning to improve training. This involved training the CNN on puzzles that would increase in difficulty. I therefore divided the dataset up into 7 categories of difficulty, followed by feeding these into the network for training in order of ascending difficulty. I also had each category overlap slightly in difficulty to help with the flow of training.

I decided to switch to a CNN since it fitted the grid structure of sudoku which is similar to images. However, this resulted in no restraints for the rules of sudoku. I therefore added a function that would introduce a penalty that would check if rows & columns had repeated digits. If so the learning rate would be reduced.

This result was overall a drastic improvement, the test accuracy for cells was 99% for the easiest puzzles and 91% for the harder puzzles. However, the puzzle accuracy was 57% for the easiest puzzles and 0.2% for the harder puzzles. This makes sense however since a correct puzzle requires all empty cells (an amount near 81) to be correct which would require a cell accuracy of essentially 100%.


## Real-Time AI Image Intensifier with Deep Learning

I built a full-stack deep learning system that transforms live webcam footage into night-vision-style enhanced video in real time.

The core: a U-Net neural network. The encoder compresses each frame through three double-convolution blocks, extracting increasingly abstract features. The decoder rebuilds the image at full resolution, with skip connections preserving fine spatial detail that would otherwise be lost. A sigmoid output layer keeps pixel values clean.

The night-vision effect is applied post-model: greyscale conversion using perceptual luminance weights, contrast normalisation, gamma correction (γ=0.6) to lift midtones, then remapping to a green-dominant RGB channel — the classic image intensifier look.

The backend captures webcam frames at 1080p, processes them through the model at reduced resolution for speed, then upscales with Lanczos interpolation. Temporal smoothing blends consecutive frames to suppress flicker, and a sharpening kernel restores edge detail.

The frontend is a browser-based client served via Flask, sending webcam frames as base64 to a REST API and displaying enhanced results in real time — accessible from any device on the local network.
