%The following function displays accuracy data about both networks to the
%console
function [] = DisplayInfo(AccuracyTrain, AccuracyTest, Type)
    %% Print the information to the console
    formatSpec = "" + ...
        "######  Network %d  Accuracy Results   ######\n" + ...
        "####     Training accuracy: %3.2f%%.     ####\n" + ...
        "####     Testing accuracy : %3.2f%%.     ####\n" + ...
        "############################################";
    sprintf(formatSpec, Type, AccuracyTrain, AccuracyTest*100)

end