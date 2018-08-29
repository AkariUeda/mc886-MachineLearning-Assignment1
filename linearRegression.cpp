#include <bits/stdc++.h>
using namespace std;
int FEATURES = 10;
int EXAMPLES = 45849;
int DOVALIDATE = 0;
int VALIDATE = 9170;
int ITER = 0;
int VERBOSE = 1;
int HOWVERBOSE = 1000;
int SGD = 0;
int RANDTHETA = 0;
int ASYNC = 0;
int MINIBATCH = 0;
int BATCHSIZE = 32;
float ALPHA = 0.00027;
float TIME  = 0;
const int MAXEXAMPLES = 50000;
const int MAXFEATURES = 10;
const int MAXVALIDATE = 10000;
const float FEXINV = (float)1/EXAMPLES;
const float DFEXINV = (float)1/(2*EXAMPLES);
const double LFEXINV = (double)1/EXAMPLES;
/*Helper functions for reading data from the .csv files*/
void read_csv(int row, int col, string filename, float dat[MAXEXAMPLES][MAXFEATURES]){
	FILE *file;
	file = fopen(filename.c_str(), "r");
	int i = 0;
    const int BUFFER = 2048;
    char line[BUFFER];
	while (fgets(line, BUFFER, file) && (i < row)){
        char* tmp = strdup(line);
	    int j = 0;
	    const char* tok;
	    for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n")){
	        dat[i][j] = atof(tok);
	    }
        free(tmp);
        i++;
    }
    return;
}
void read_array(int row, string filename, float dat[]){
	FILE *file;
	file = fopen(filename.c_str(), "r");
	int i = 0;
    const int BUFFER = 2048;
    char line[BUFFER];
	while(fgets(line, BUFFER, file) && (i < row)){
        char* tmp = strdup(line);
	    dat[i] = atof(tmp);
        free(tmp);
        i++;
    }
    return;
}


vector<string> split(const string& s, char delimiter){
   vector<string> tokens;
   string token;
   istringstream tokenStream(s);
   while (getline(tokenStream, token, delimiter)) tokens.push_back(token);
   return tokens;
}
template<class T> T strToNum(const string s){
    stringstream ss;
    ss << s;
    T v;
    ss >> v;
    return v;
}
/*Helper functions for matrix manipulation*/
void transpose(float A[MAXEXAMPLES][MAXFEATURES], float B[MAXFEATURES][MAXEXAMPLES]){
    for(int i = 0;i<FEATURES;i++){
        for(int j = 0;j<EXAMPLES;j++){
            B[i][j] = A[j][i];
        }
    }
    return;
}
float** allocateMatrix(int lin, int col){
    float **m;
    m = (float**) malloc(lin*sizeof(float*));
    if(m == NULL) exit(0);
    for(int i = 0;i<lin;i++){
        m[i] = (float*)malloc(col*sizeof(float));
        if(m[i] == NULL) exit(0);
    }
    return m;
}
float** freeMatrix(int lin, int col,float **m){
    for(int i = 0;i<lin;i++) free(m[i]);
    free(m);
    return NULL;
}
void matMult(float **A,int la,int lb,float **B,int ca,int cb,float **C){
    for(int i = 0;i<la;i++){
        for(int j = 0;j<cb;j++){
            C[i][j] = 0;
            for(int k = 0;k  < ca;k++) C[i][j] += A[i][k] + B[k][j];
        }
    }
}

/*Helper functions to calculate hypothesis, summations and cost for the regressions*/
float h(float x[],float theta[]){
    float sum = 0.0, c = 0.0;
    for(int i = 0;i<FEATURES;i++){ 
        float y = (theta[i]*x[i]) - c;
        float t = sum + y;
        c = (t-sum)-y;
        sum = t;
    }    
    return sum;
}
float summation(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xj[],float theta[],int ini, int fim){
    float sum = 0.0, c = 0.0;
    for(int i = ini;i<fim && i<EXAMPLES;i++){
        float yl = (h(x[i],theta)-y[i])*xj[i] - c;
        float t = sum + yl;
        c = (t-sum)-yl;
        sum = t;
    }
    return sum;
}
float cost(float theta[],float y[],float x[MAXEXAMPLES][MAXFEATURES]){
    float sum = 0.0, c = 0.0;
    for(int i = 0;i<EXAMPLES;i++){
        float val = (h(x[i],theta)-y[i]);
        float yl = val*val - c;
        float t = sum + yl;
        c = (t-sum)-yl;
        sum = t;
    }
    return sum/(2*EXAMPLES);
}
void randomTheta(float theta[]){
	for(int i = 0;i<FEATURES;i++) theta[i] = (rand()&1 ? -1 : 1)*rand()%1663; 
}
bool isNan(float val){
	return val != val || val > FLT_MAX || val < -FLT_MAX;
}
/*Helper functions that print and log values*/
void printV(float v[], int T){
    for(int i = 0;i<T;i++) printf("%f ",v[i]);
    printf("\n");
    return;
}
void predict(float x[MAXEXAMPLES][MAXFEATURES],float y[],float theta[]){
    FILE *fp = fopen("predictions.csv", "w+");
	for(int i = 0;i<VALIDATE;i++){
		char s[2048];
		sprintf(s,"\"%f\",\"%f\"\n",h(x[i],theta),y[i]);   
    	fputs(s,fp );      
    }
    printf("Custo predito: %f\n",cost(theta,y,x));
    fclose(fp );
}
void writeInfo(FILE *fp,float cus,int i,bool isPred){
	char s[2048];
	if(i == 1) sprintf(s,"\"%f\"",cus);   
	else sprintf(s,",\"%f\"",cus);   
	fputs(s,fp);      
	if(VERBOSE && i%HOWVERBOSE == 0){
	    if(isPred)
	        printf("Iteracao %6d: custo predito de %f\n",i,cus); 
	    else
	        printf("Iteracao %6d: custo de %f\n",i,cus);
	}  
}
void writeTheta(float theta[]){
	FILE *th = fopen("theta.csv", "w+");
    for(int i = 0;i<FEATURES;i++){
    	float val = theta[i];  
        char s[2048];
		sprintf(s,"\"%f\",",val);   
    	fputs(s,th);   
	}	
    fclose(th);
}
bool writeCostToFile(FILE *fp,FILE *fpPred,FILE *fpTime,FILE *fpTimePred,float ts,float theta[],float x[MAXEXAMPLES][MAXFEATURES],float y[],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],int i){
    float cus = cost(theta,y,x);
    writeInfo(fp,cus,i,false);
    if(TIME)
        writeInfo(fpTime,ts,i,false);
    if(DOVALIDATE){
        float cusPred = cost(theta,yVal,xVal);
        writeInfo(fpPred,cusPred,i,true);
        if(TIME)
            writeInfo(fpTimePred,ts,i,false);
    }  
    return true;
}
void closeFiles(FILE *fp,FILE *fpPred,float theta[],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[]){
    fclose(fp);
    if(DOVALIDATE)
	    fclose(fpPred);
    writeTheta(theta);
    if(VERBOSE) for(int i = 0;i<FEATURES;i++) printf("Theta %d = %f\n",i,theta[i]);
    if(DOVALIDATE && VERBOSE)
        predict(xVal,yVal,theta);
}

/*Gradient descent functions */
void gradientDescBatchAsync(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
	bool stop = false;    
	float ts = 0;
    for(int i = 1;i<=ITER && !stop;i++){
    	future<float> hold[FEATURES];
        for(int j = 0;j<FEATURES;j++){
            hold[j] = async(launch::async,summation,x,y,xt[j],theta,0,EXAMPLES);
        }        
        for(int j = 0;j<FEATURES && !stop;j++){
        	float sum = hold[j].get();
        	theta[j] = theta[j] - (ALPHA*sum)/EXAMPLES;
        	stop = isNan(theta[j]);
		}     
        writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,i);
    }
}
void gradientDescBatch(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
   
    float oldTheta[FEATURES];  
    float ts = 0;  
	bool stop = false; 
    for(int i = 1;i<=ITER && !stop;i++){
        for(int j = 0;j<FEATURES && !stop;j++){
            float sum = summation(x,y,xt[j],theta,0,EXAMPLES);
            oldTheta[j] = theta[j] - (ALPHA*sum)/EXAMPLES;
			stop = isNan(oldTheta[j]);
        }        
        memcpy(theta,oldTheta,FEATURES*sizeof(float));  
        writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,i);
          
    }
}
void gradientDescStochastic(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    float oldTheta[FEATURES];   
    float ts = 0;
	bool stop = false;  
    for(int i = 1;i<=ITER && !stop;i++){        
        float func = h(x[(i-1)%EXAMPLES],theta);
        for(int j = 0;j<FEATURES && !stop;j++){            
            oldTheta[j] = theta[j] - ALPHA*(func-y[(i-1)%EXAMPLES])*xt[j][(i-1)%EXAMPLES];  
			stop = isNan(oldTheta[j]); 
        }        
        memcpy(theta,oldTheta,FEATURES*sizeof(float));
        writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,i);          
    }
}
void gradientDescMiniBAsync(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    int masterIt = 1; 
    bool stop = false;
    float ts = 0;
    for(int i = 1;i<=ITER && !stop;i++){
        for(int b = 0;b<EXAMPLES && !stop;b+=BATCHSIZE){
        	future<float> hold[FEATURES];
            for(int j = 0;j<FEATURES;j++){
                hold[j] = async(launch::async,summation,x,y,xt[j],theta,b,b+BATCHSIZE-1);
            }        
            for(int j = 0;j<FEATURES && !stop;j++){
            	float sum = hold[j].get();
            	theta[j] = theta[j] - (ALPHA*sum)/BATCHSIZE;
            	stop = isNan(theta[j]);
		    }		        
            writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,masterIt);
            masterIt++;
        }
    }
}
void gradientDescMiniB(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    float oldTheta[FEATURES];
    float ts = 0;
    int masterIt = 1;  
    bool stop = false;
    for(int i = 1;i<=ITER && !stop;i++){
        for(int b = 0;b<EXAMPLES && !stop;b+=BATCHSIZE){
            for(int j = 0;j<FEATURES && !stop;j++){
                float sum = summation(x,y,xt[j],theta,b,b+BATCHSIZE-1);
                oldTheta[j] = theta[j] - (ALPHA*sum)/BATCHSIZE;
                stop = isNan(oldTheta[j]);
            }        
            memcpy(theta,oldTheta,FEATURES*sizeof(float));  
            writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,masterIt);         
			masterIt++;       
        }
    }
}
/*Gradient descent timed functions */
void gradientDescBatchAsyncTimed(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
	int i = 1; 
    auto Start = chrono::high_resolution_clock::now(); 
    bool stop = false;
    float ts = 0;
    while(!stop){        
    	future<float> hold[FEATURES];
        for(int j = 0;j<FEATURES;j++){
            hold[j] = async(launch::async,summation,x,y,xt[j],theta,0,EXAMPLES);
        }        
        for(int j = 0;j<FEATURES && !stop;j++){
        	float sum = hold[j].get();
        	theta[j] = theta[j] - (ALPHA*sum)/BATCHSIZE;
        	stop = isNan(theta[j]);
		}  
		auto End = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> Elapsed = End - Start;
		if (Elapsed.count() >= TIME)
			break;
	    ts = Elapsed.count();   
        writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,i);
        i++;
    }
}
void gradientDescBatchTimed(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    float oldTheta[FEATURES];    
	int i = 0; 
	float ts = 0;
	bool stop = false;
    auto Start = chrono::high_resolution_clock::now(); 
    while(!stop){
      
        for(int j = 0;j<FEATURES && !stop;j++){
            float sum = summation(x,y,xt[j],theta,0,EXAMPLES);
            oldTheta[j] = theta[j] - (ALPHA*sum)/BATCHSIZE;
            stop = isNan(oldTheta[j]);
        }        
        memcpy(theta,oldTheta,FEATURES*sizeof(float));  
        auto End = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> Elapsed = End - Start;
		if (Elapsed.count() >= TIME)
			break;
	    ts = Elapsed.count();   
        writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,i);
		i++;         
    }
}
void gradientDescStochasticTimed(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    float oldTheta[FEATURES];    
    auto Start = chrono::high_resolution_clock::now(); 
    int i = 1;
    float ts = 0;
    bool stop = false;
    while(!stop){
       
        float func = h(x[i%EXAMPLES],theta);
        for(int j = 0;j<FEATURES && !stop;j++){            
            oldTheta[j] = theta[j] - ALPHA*(func-y[(i-1)%EXAMPLES])*xt[j][(i-1)%EXAMPLES];  
			stop = isNan(oldTheta[j]); 
        }        
        memcpy(theta,oldTheta,FEATURES*sizeof(float));
        auto End = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> Elapsed = End - Start;
		if (Elapsed.count() >= TIME)
			break;
	    ts = Elapsed.count();
        writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,i);
        i++;
    }
}
void gradientDescMiniBAsyncTimed(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    float ts;
    int masterIt = 1; 
    bool stop = false;
    auto Start = chrono::high_resolution_clock::now(); 
    while(!stop){
        auto End = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> Elapsed = End - Start;
        if (Elapsed.count() >= TIME)
	        break;
        ts = Elapsed.count();
        for(int b = 0;b<EXAMPLES && !stop;b+=BATCHSIZE){
        	future<float> hold[FEATURES];
            for(int j = 0;j<FEATURES;j++){
                hold[j] = async(launch::async,summation,x,y,xt[j],theta,b,b+BATCHSIZE-1);
            }        
            for(int j = 0;j<FEATURES && !stop;j++){
            	float sum = hold[j].get();
            	theta[j] = theta[j] - (ALPHA*sum)/BATCHSIZE;
            	stop = isNan(theta[j]);
	        }		
	        auto End2 = chrono::high_resolution_clock::now();
	        chrono::duration<double, milli> Elapsed2 = End2 - Start;
	        if (Elapsed2.count() >= TIME)
		        break;
		    ts = Elapsed2.count();        
            writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,masterIt);            
		    masterIt++;
        }
    }
}

void gradientDescMiniBTimed(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[], FILE *costCsv, FILE *predictCsv,FILE *fpTime,FILE *fpTimePred){
    float oldTheta[FEATURES];
    int masterIt = 1;  
    bool stop = false;
    float ts = 0;
    auto Start = chrono::high_resolution_clock::now(); 
    while(!stop){
        auto End = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> Elapsed = End - Start;
		if (Elapsed.count() >= TIME)
			break;
		ts = Elapsed.count();
        for(int b = 0;b<EXAMPLES && !stop;b+=BATCHSIZE){
            for(int j = 0;j<FEATURES && !stop;j++){
                float sum = summation(x,y,xt[j],theta,b,b+BATCHSIZE-1);
                oldTheta[j] = theta[j] - (ALPHA*sum)/EXAMPLES;
                stop = isNan(oldTheta[j]);                
            }        
            memcpy(theta,oldTheta,FEATURES*sizeof(float));
            auto End2 = chrono::high_resolution_clock::now();
		    chrono::duration<double, milli> Elapsed2 = End2 - Start;
		    if (Elapsed2.count() >= TIME)
			    break;  
			ts = Elapsed2.count();
            writeCostToFile(costCsv,predictCsv,fpTime,fpTimePred,ts,theta,x,y,xVal,yVal,masterIt);
           
			masterIt++;
        }
    }
}
/*Main*/
int main(int argc, char** argv){
    float traindata[MAXEXAMPLES][MAXFEATURES],label[MAXEXAMPLES],dataTransp[MAXFEATURES][MAXEXAMPLES],dataVal[MAXVALIDATE][MAXFEATURES],labelVal[MAXVALIDATE],theta[FEATURES];       
   
	srand(time(NULL));
    const string HELP = "-features ou -f          : define o numero de features (10 por padrão)\n-examples ou -e          : define o numero de exemplos pra treino (45849 por padrão)\n-validates ou -v         : define o numero de exemplos pra validacao (9170 por padrão)\n-iterations ou -i        : define o numero de iteracoes da regressão (1000 por padrão)\n-alpha ou -a             : define o valor da learning rate (0.00027 por padrão)\n-verbose ou -vr          : imprime ou nao os resultados a cada N iteracoes (0 desligado, !0 ligado, ligado por padrão)\n-stochasticdesc ou -sgd  : faz stochastic gradient descent no lugar de batch gradient descent (0 desligado, !0 ligado, desligado por padrão)\n-randtheta ou -rt        : inicializa o vetor de thetas com valores aleatórios (0 desligado, !0 ligado, desligado por padrão)\n-howverbose ou -hvr      : define a cada quantas iterações devem ser impressos os resultados (1000 por padrão)\n-trainfeatures ou -tf    : indica o nome do arquivo com as features para treino (train_features.csv por padrão)\n-trainlabels ou -tl      : indica o nome do arquivo com as labels para treino (train_labels.csv por padrão)\n-validatefeatures ou -vf : indica o nome do arquivo com as features para validação (valid_features.csv por padrão)\n-validatelabels ou -vl   : indica o nome do arquivo com as labels para validação (valid_labels.csv por padrão)\n-help ou -h              : exibe este texto e termina\n";
    
	
	string fnameEx = "train_features.csv",fnameLabel= "train_labels.csv",fnameVal= "valid_features.csv",fnameValLabel = "valid_labels.csv";
    for(int i = 1;i<argc;i++){
        vector<string> args = split(string(argv[i]),'=');
        if(args[0] == "-features" || args[0] == "-f") FEATURES = strToNum<int>(args[1]);
        else if(args[0] == "-examples" || args[0] == "-e") EXAMPLES = strToNum<int>(args[1]);
        else if(args[0] == "-validates" || args[0] == "-v") VALIDATE = strToNum<int>(args[1]);
        else if(args[0] == "-iterations" || args[0] == "-i") ITER = strToNum<int>(args[1]);
        else if(args[0] == "-verbose" || args[0] == "-vr") VERBOSE = strToNum<int>(args[1]);
        else if(args[0] == "-howverbose" || args[0] == "-hvr") HOWVERBOSE = strToNum<int>(args[1]);
        else if(args[0] == "-trainfeatures" || args[0] == "-tf") fnameEx = args[1];
        else if(args[0] == "-trainlabels" || args[0] == "-tl") fnameLabel = args[1];
        else if(args[0] == "-dovalidate" || args[0] == "-dvl") DOVALIDATE = strToNum<int>(args[1]);
        else if(args[0] == "-validatefeatures" || args[0] == "-vf") fnameVal = args[1];
        else if(args[0] == "-validatelabels" || args[0] == "-vl") fnameValLabel = args[1];
        else if(args[0] == "-alpha" || args[0] == "-a") ALPHA = strToNum<float>(args[1]);
        else if(args[0] == "-stochasticdesc" || args[0] == "-sgd") SGD = strToNum<int>(args[1]);
        else if(args[0] == "-randtheta" || args[0] == "-rt") RANDTHETA = strToNum<int>(args[1]);
        else if(args[0] == "-minibatch" || args[0] == "-mb") MINIBATCH = strToNum<int>(args[1]);
        else if(args[0] == "-timed" || args[0] == "-time") TIME = strToNum<float>(args[1]);
        else if(args[0] == "-batchsize" || args[0] == "-bs") BATCHSIZE = strToNum<int>(args[1]);
        else if(args[0] == "-async" || args[0] == "-as") ASYNC = strToNum<int>(args[1]);
        else if(args[0] == "-help" || args[0] == "-h"){
            printf("%s",HELP.c_str());
            return 0;
        }
    }
	FILE *costCsv = fopen("costs.csv", "w+"), *timeCsv = fopen("times.csv", "w+");
    FILE *costPredCsv, *timePredCsv;
    if(DOVALIDATE){
        costPredCsv = fopen("predictCosts.csv", "w+");
	    timePredCsv = fopen("predictTimes.csv", "w+");
	}
    if(FEATURES <= 0){printf("Por favor, use um número positivo de features!\n");return 0;}
    if(EXAMPLES <= 0){printf("Por favor, use um número positivo de exemplos de treino!\n");return 0;}
    if(VALIDATE <= 0){printf("Por favor, use um número positivo de exemplos de validacao!\n");return 0;}
    if(HOWVERBOSE <= 0){printf("É impossível imprimir resultados a cada %d iterações!\n",HOWVERBOSE);return 0;}
	int row = EXAMPLES;
	int col = FEATURES;
	if(RANDTHETA) randomTheta(theta);

	read_csv(row, col, fnameEx, traindata);
	read_array(row,fnameLabel,label);	
	read_csv(row, col, fnameVal, dataVal);
	read_array(row,fnameValLabel,labelVal);
	transpose(traindata,dataTransp);
	TIME *= 1000;
	if(MINIBATCH){
        if(!ASYNC)
        	if(!TIME)
        		gradientDescMiniB(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
        	else
        		gradientDescMiniBTimed(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
		else
        	if(!TIME)
        		gradientDescMiniBAsync(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
        	else
        		gradientDescMiniBAsyncTimed(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
    }else if(SGD){
        if(!TIME)
        	gradientDescStochastic(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
        else
        	gradientDescStochasticTimed(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);    
    }else{
        if(!ASYNC)
        	if(!TIME)
        		gradientDescBatch(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
        	else
        		gradientDescBatchTimed(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
		else
        	if(!TIME)
        		gradientDescBatchAsync(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
        	else
        		gradientDescBatchAsyncTimed(traindata,label,dataTransp,dataVal,labelVal,theta,costCsv,costPredCsv,timeCsv,timePredCsv);
    }
    
    closeFiles(costCsv,costPredCsv,theta,dataVal,labelVal);
	return 0;
}
