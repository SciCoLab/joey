#include<stdio.h>


void convolute(double output [3][3], double input[5][5], double kernel[3][3] )
{
    int convolute = 0; // This holds the convolution results for an index.
    int x, y; // Used for input matrix index

	// Fill output matrix: rows and columns are i and j respectively
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			x = i;
			y = j;

			// Kernel rows and columns are k and l respectively
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					// Convolute here.
					convolute += kernel[k][l] * input[x][y];
					y++; // Move right.
				}
				x++; // Move down.
				y = j; // Restart column position
			}
			output[i][j] = convolute; // Add result to output matrix.
			convolute = 0; // Needed before we move on to the next index.
		}
	}
}

void main(int argc, char * argv[]){


double input[5][5];

int i =0; int j=0;

double result [3][3]; 

int c =0;
for(i=0;i<5;i++){


    for(j=0;j<5;j++){

        input[i][j] = c;
        printf("%f", input[i][j]);
        printf("\t");
        c++;
    } 
    printf("\n");

} 

double kernel [3][3] = {{0,0,0} ,{1,0,-1},{0,0,0}};

convolute(result, input, kernel);


for(i=0;i<3;i++){


    for(j=0;j<3;j++){

        printf("%f", result[i][j]);
        
    } 

    printf("\n");

} 


}
