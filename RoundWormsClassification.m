% worm analysis 

% pretrained network initializing 
anet = alexnet ;

% creating datastore
imgs            = imageDatastore("WormImages", ...
                                  "IncludeSubfolders", true ) ;
                  
% getting correct image labels                  
trueLabels      = readtable("WormData.csv") ;
imgs.Labels     = categorical(trueLabels.Status) ;

% dividing data into test and train datasets
[trainimg, testimg] = splitEachLabel(imgs, ...
                                     0.7, "randomized" );

% adding fully connected layer for better training and prediction
layers          = anet.Layers ; 
fc              = fullyConnectedLayer(2) ;
layers(end-2)   = fc ;
% adding classification layer at the end
cl              = classificationLayer ;
layers(end)     = cl ;

% preprocessing images
trainds         = augmentedImageDatastore([227,227] , trainimg , ...
                                          "ColorPreprocessing" , "gray2rgb" ) ;
testds          = augmentedImageDatastore([227,227] , testimg , ...
                                          "ColorPreprocessing" , "gray2rgb" ) ;
% setting training options
opts            = trainingOptions( "sgdm" , ...
                                   "InitialLearnRate" , 0.001, ...
                                   "Momentum" , 0.4, ...
                                   "MaxEpochs" , 60 , ...
                                   "Plots" , "training-progress" );

% training the network
[net, info]     = trainNetwork( trainds , layers , opts ) ; 
% making predictions
preds           = classify( net , testds ) ;
testActual      = testimg.Labels ;
% comparison and evaluation of the result
acc             = nnz( preds == testActual ) / size( testActual , 1 ) 
conf            = confusionchart( testActual , preds ) ;