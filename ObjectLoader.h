#include "VectorFunctions.h"

void ReadOBJFile()
{
	FILE* fp;
	fp = fopen("OBJ_Files/Room1.obj","r");
	int cnt = 0;

	if(fp != NULL)
	{
		bool isVn = false, isVt = false;
        char ch;

		int objectCount = 0;
		Vec3 objectColor = Vector3(250.0, 250.0, 250.0);
		float objectReflectivity;
		Vec3 p;
		p.x = 0;
		p.y = 0;
		p.z = 0;

		sceneBB.minP.x = 99999;
		sceneBB.minP.y = 99999;
		sceneBB.minP.z = 99999;

		sceneBB.maxP.x = -99999;
		sceneBB.maxP.y = -99999;
		sceneBB.maxP.z = -99999;

		while((ch=fgetc(fp))!= EOF)
		{
			cnt++;
			if(ch == '\n')
				continue;
			if(ch == 'o')
			{
				ch = fgetc(fp);
				
				if (ch == ' ')
				{
					objectCount++;

					printf("Enter Color for Object %d: \n", objectCount);

					printf("R: ");
					scanf("%f", &objectColor.x);
					printf("G: ");
					scanf("%f", &objectColor.y);
					printf("B: ");
					scanf("%f", &objectColor.z);

					printf("Enter Reflectivity of Object %d: \n", objectCount);

					scanf("%f", &objectReflectivity);
				}

			}
            else if(ch == 'v')
            {
                ch=fgetc(fp);
			   
                if(ch=='t')
                {
				  isVt = true;
                  fscanf(fp," %f %f\n",&p.x,&p.y);
                  vt.push_back(p);
                }
                else if(ch=='n')
                {
					isVn = true;
                    fscanf(fp," %f %f %f\n",&p.x,&p.y,&p.z);
                    vn.push_back(p);
                }
                else
                {
                    fscanf(fp,"%f %f %f\n",&p.x,&p.y,&p.z);
					
					p = Scale(p, 100.0, 100.0, 30.0);
					p = Translate(p, 100.0, -300.0, -100.0);

					if(p.x > sceneBB.maxP.x)
						sceneBB.maxP.x = p.x;

					if(p.y > sceneBB.maxP.y)
						sceneBB.maxP.y = p.y;

					if(p.z > sceneBB.maxP.z)
						sceneBB.maxP.z = p.z;

					if(p.x < sceneBB.minP.x)
						sceneBB.minP.x = p.x;

					if(p.y < sceneBB.minP.y)
						sceneBB.minP.y = p.y;

					if(p.z < sceneBB.minP.z)
						sceneBB.minP.z = p.z;

                    v.push_back(p);
                }
            }
            else if(ch == 'f')
            {
				 ch = fgetc(fp);
				 Face ftemp;

				 ftemp.minBB.x = 9999;
				 ftemp.minBB.y = 9999;
				 ftemp.minBB.z = 9999;


				 ftemp.maxBB.x = -9999;
				 ftemp.maxBB.y = -9999;
				 ftemp.maxBB.z = -9999;

				 //char c[2048];
				
				 int vindex = -1;
				 int vtindex = -1;
				 int vnindex = -1;

				 int vindexVal = 0;
				 int vtindexVal = 0;
				 int vnindexVal = 0;

				 do
				 {
					 if(fscanf(fp,"%d", &vindexVal) != 0)
					 {
						 vindex++;
						 ch = fgetc(fp);

						 if(isVt)
						 {
							if(ch == '/')
							{
								if(fscanf(fp,"%d", &vtindexVal) != 0)
								{
									ch = fgetc(fp);
									vtindex++;
								}
							}
						 }
						 if(isVn)
						 {
							if(ch == '/')
							{
								ch = fgetc(fp);
								if(fscanf(fp,"%d", &vnindexVal) != 0)
						   		{
									ch = fgetc(fp);
									vnindex++;
								}
							 }
						 }
					 }
					 else
					 {
						 break;
					 }
					
					 ftemp.v[vindex] = v[vindexVal-1];
					 //if(isVt)
						//ftemp.vt[vtindex] = vt[vtindexVal-1];
					 //if(isVn)
						//ftemp.vn[vnindex] = vn[vnindexVal-1];

					 if( ftemp.minBB.x > ftemp.v[vindex].x)
					 {
						 ftemp.minBB.x = ftemp.v[vindex].x;
					 }
					 if( ftemp.minBB.y > ftemp.v[vindex].y)
					 {
						 ftemp.minBB.y = ftemp.v[vindex].y;
					 }
					 if( ftemp.minBB.z > ftemp.v[vindex].z)
					 {
						 ftemp.minBB.z = ftemp.v[vindex].z;
					 }

					 if( ftemp.maxBB.x < ftemp.v[vindex].x)
					 {
						 ftemp.maxBB.x = ftemp.v[vindex].x;
					 }
					 if( ftemp.maxBB.y < ftemp.v[vindex].y)
					 {
						 ftemp.maxBB.y = ftemp.v[vindex].y;
					 }
					 if( ftemp.maxBB.z < ftemp.v[vindex].z)
					 {
						 ftemp.maxBB.z = ftemp.v[vindex].z;
					 }
				 }while(ch != '\n');

				 Vec3 edge0 = Sub(ftemp.v[0], ftemp.v[1]);
				 Vec3 edge1 = Sub(ftemp.v[0], ftemp.v[2]);

				 ftemp.normalF = Normalize(CrossProduct(edge0, edge1));
				 ftemp.color = objectColor;
				 ftemp.reflectivity = objectReflectivity;

				 f_host.push_back(ftemp);
            }
			else
			{
				char c[1024];
				fscanf(fp,"%[^\n]\n", c);
			}
		}
		v.clear();
		vt.clear();
		vn.clear();
	}
	else
	{
		printf("File has not opened successfully");
	}
	
	fclose(fp);
}