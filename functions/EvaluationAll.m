function ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
ResultAll=zeros(15,1);

HammingLoss=Hamming_loss(Pre_Labels,test_target);
RankingLoss=Ranking_loss(Outputs,test_target);
OneError=One_error(Outputs,test_target);
Average_Precision=Average_precision(Outputs,test_target);

ResultAll(1,1)=HammingLoss;
ResultAll(12,1)=Average_Precision;
ResultAll(13,1)=OneError;
ResultAll(14,1)=RankingLoss;

end