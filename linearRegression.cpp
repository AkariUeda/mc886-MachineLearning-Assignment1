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
float ALPHA = 0.00027;
const int MAXEXAMPLES = 50000;
const int MAXFEATURES = 10;
const int MAXVALIDATE = 10000;
const float FEXINV = (float)1/EXAMPLES;
const float DFEXINV = (float)1/(2*EXAMPLES);
const double LFEXINV = (double)1/EXAMPLES;
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
float summation(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xj[],float theta[]){
    float sum = 0.0, c = 0.0;
    for(int i = 0;i<EXAMPLES;i++){
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
void gradientDescBatchAsync(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[]){
    FILE *fp = fopen("costs.csv", "w+");
    FILE *fpPred;
    if(DOVALIDATE)
         fpPred = fopen("predictCosts.csv", "w+");
    float alpha = 0.00027;
    if(RANDTHETA)
        for(int i = 0;i<FEATURES;i++) theta[i] = rand()%1663;     
    for(int i = 1;i<=ITER;i++){
    	future<float> hold[FEATURES];
        for(int j = 0;j<FEATURES;j++){
            hold[j] = async(launch::async,summation,x,y,xt[j],theta);
        }        
        for(int j = 0;j<FEATURES;j++){
        	float sum = hold[j].get();
        	theta[j] = theta[j] - (ALPHA*sum)/EXAMPLES;
		}
        
        float cus = cost(theta,y,x);
        writeInfo(fp,cus,i,false);
        if(DOVALIDATE){
            float cusPred = cost(theta,yVal,xVal);
            writeInfo(fpPred,cusPred,i,true);
        }
        if(cus != cus || cus > FLT_MAX || cus < -FLT_MAX){
            break;
        }
    }
    fclose(fp);
    if(DOVALIDATE)
	    fclose(fpPred);
    writeTheta(theta);
    if(VERBOSE) for(int i = 0;i<FEATURES;i++) printf("Theta %d = %f\n",i,theta[i]);
    if(DOVALIDATE && VERBOSE)
        predict(xVal,yVal,theta);
}
void gradientDescBatch(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[],float theta[]){
    FILE *fp = fopen("costs.csv", "w+");
    FILE *fpPred;
    if(DOVALIDATE)
         fpPred = fopen("predictCosts.csv", "w+");
    float oldTheta[FEATURES];
    if(RANDTHETA)
        for(int i = 0;i<FEATURES;i++) theta[i] = rand()%1663;     
    for(int i = 1;i<=ITER;i++){
        for(int j = 0;j<FEATURES;j++){
            float sum = summation(x,y,xt[j],theta);
            oldTheta[j] = theta[j] - (ALPHA*sum)/EXAMPLES;
        }        
        memcpy(theta,oldTheta,FEATURES*sizeof(float));  
        float cus = cost(theta,y,x);        
        writeInfo(fp,cus,i,false);
        if(DOVALIDATE){
            float cusPred = cost(theta,yVal,xVal);
            writeInfo(fpPred,cusPred,i,true);
        }
        if(cus != cus || cus > FLT_MAX || cus < -FLT_MAX){
            break;
        }
    }
	fclose(fp);
	if(DOVALIDATE)
	    fclose(fpPred);
	writeTheta(theta);
    if(VERBOSE) for(int i = 0;i<FEATURES;i++) printf("Theta %d = %f\n",i,theta[i]);
    if(DOVALIDATE && VERBOSE)
        predict(xVal,yVal,theta);
}
void gradientDescStochastic(float x[MAXEXAMPLES][MAXFEATURES],float y[],float xt[MAXFEATURES][MAXEXAMPLES],float xVal[MAXVALIDATE][MAXFEATURES],float yVal[]){
    FILE *fp = fopen("costs.csv", "w+");    
    float oldTheta[FEATURES], theta[FEATURES] = {0};
    if(RANDTHETA)
        for(int i = 0;i<FEATURES;i++) theta[i] = rand()%1663;     
    for(int i = 1;i<=ITER;i++){
        for(int m = 0;m<EXAMPLES;m++){
            for(int j = 0;j<FEATURES;j++){            
                oldTheta[j] = theta[j] - ALPHA*(h(x[m],theta)-y[m])*xt[j][m];   
            }        
            memcpy(theta,oldTheta,FEATURES*sizeof(float));
        }            
        writeInfo(fp,cost(theta,y,x),i,true);
    }
    fclose(fp);
    writeTheta(theta);
    if(VERBOSE) for(int i = 0;i<FEATURES;i++) printf("Theta %d = %f\n",i,theta[i]);
    if(DOVALIDATE && VERBOSE)
        predict(xVal,yVal,theta);
}
void transpose(float A[MAXEXAMPLES][MAXFEATURES], float B[MAXFEATURES][MAXEXAMPLES]){
    for(int i = 0;i<FEATURES;i++){
        for(int j = 0;j<EXAMPLES;j++){
            B[i][j] = A[j][i];
        }
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
        else if(args[0] == "-async" || args[0] == "-as") ASYNC = strToNum<int>(args[1]);
        else if(args[0] == "-help" || args[0] == "-h"){
            printf("%s",HELP.c_str());
            return 0;
        }
    }
    
    if(FEATURES <= 0){printf("Por favor, use um número positivo de features!\n");return 0;}
    if(EXAMPLES <= 0){printf("Por favor, use um número positivo de exemplos de treino!\n");return 0;}
    if(VALIDATE <= 0){printf("Por favor, use um número positivo de exemplos de validacao!\n");return 0;}
    if(HOWVERBOSE <= 0){printf("É impossível imprimir resultados a cada %d iterações!\n",HOWVERBOSE);return 0;}
	int row = EXAMPLES;
	int col = FEATURES;
	
	read_csv(row, col, fnameEx, traindata);
	read_array(row,fnameLabel,label);	
	read_csv(row, col, fnameVal, dataVal);
	read_array(row,fnameValLabel,labelVal);
	transpose(traindata,dataTransp);
	if(!SGD){
		if(!ASYNC)
        	gradientDescBatch(traindata,label,dataTransp,dataVal,labelVal,theta);
        else
        	gradientDescBatchAsync(traindata,label,dataTransp,dataVal,labelVal,theta);        	
    }else{
        gradientDescStochastic(traindata,label,dataTransp,dataVal,labelVal);
    }
	return 0;
}
