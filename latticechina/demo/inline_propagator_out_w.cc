/*! \file
 * \brief Inline construction of propagator
 *
 * Propagator calculations
 */
#include<complex>
#include<vector>
#include<math.h>
#include<stdio.h>
#include "fermact.h"
#include "inline_propagator_out_w.h"
#include "meas/inline/abs_inline_measurement_factory.h"
#include "meas/glue/mesplq.h"
#include "util/ft/sftmom.h"
#include "util/info/proginfo.h"
#include "util/info/unique_id.h"
#include "actions/ferm/fermacts/fermact_factory_w.h"
#include "actions/ferm/fermacts/fermacts_aggregate_w.h"
#include "meas/inline/make_xml_file.h"
#include "meas/inline/io/named_objmap.h"
#include "util/ferm/transf.h"
#include "meas/hadron/BuildingBlocks_w.h"
//#include "qdp_util.h"

//#include "prop_and_gauge.h"
//#include "prop_and_gauge.h"
#include "/public/home/sunpeng/LRB/dslash_check/latticechina/inc/dslash.h"
using namespace std;

namespace Chroma 
{ 
  namespace InlinePropagator2Env
  { 
    namespace
    {
      AbsInlineMeasurement* createMeasurement(XMLReader& xml_in, 
					      const std::string& path) 
      {
	return new InlinePropagator2(InlinePropagator2Params(xml_in, path));
      }

      //! Local registration flag
      bool registered = false;
    }

    const std::string name = "PROPAGATOR_out";

    //! Register all the factories
    bool registerAll() 
    {
      bool success = true; 
      if (! registered)
      {
	success &= WilsonTypeFermActsEnv::registerAll();
	success &= TheInlineMeasurementFactory::Instance().registerObject(name, createMeasurement);
	registered = true;
      }
      return success;
    }
  } // end namespace


  //! Propagator input
  void read(XMLReader& xml, const std::string& path, InlinePropagator2Params::NamedObject_t& input)
  {
    XMLReader inputtop(xml, path);

    read(inputtop, "gauge_id", input.gauge_id);
    read(inputtop, "source_id", input.source_id);
    read(inputtop, "prop_id", input.prop_id);
  }

  //! Propagator output
  void write(XMLWriter& xml, const std::string& path, const InlinePropagator2Params::NamedObject_t& input)
  {
    push(xml, path);

    write(xml, "gauge_id", input.gauge_id);
    write(xml, "source_id", input.source_id);
    write(xml, "prop_id", input.prop_id);

    pop(xml);
  }


  // Param stuff
  InlinePropagator2Params::InlinePropagator2Params() { frequency = 0; }

  InlinePropagator2Params::InlinePropagator2Params(XMLReader& xml_in, const std::string& path) 
  {
    try 
    {
      XMLReader paramtop(xml_in, path);

      if (paramtop.count("Frequency") == 1)
	read(paramtop, "Frequency", frequency);
      else
	frequency = 1;

      // Parameters for source construction
      read(paramtop, "Param", param);

      // Read in the output propagator/source configuration info
      read(paramtop, "NamedObject", named_obj);

      // Possible alternate XML file pattern
      if (paramtop.count("xml_file") != 0) 
      {
	read(paramtop, "xml_file", xml_file);
      }
    }
    catch(const std::string& e) 
    {
      QDPIO::cerr << __func__ << ": Caught Exception reading XML: " << e << std::endl;
      QDP_abort(1);
    }
  }


  void
  InlinePropagator2Params::writeXML(XMLWriter& xml_out, const std::string& path) 
  {
    push(xml_out, path);
    
    write(xml_out, "Param", param);
    write(xml_out, "NamedObject", named_obj);

    pop(xml_out);
  }


  // Function call
  void 
  InlinePropagator2::operator()(unsigned long update_no,
			       XMLWriter& xml_out) 
  {
    // If xml file not empty, then use alternate
    if (params.xml_file != "")
    {
      std::string xml_file = makeXMLFileName(params.xml_file, update_no);

      push(xml_out, "propagator");
      write(xml_out, "update_no", update_no);
      write(xml_out, "xml_file", xml_file);
      pop(xml_out);

      XMLFileWriter xml(xml_file);
      func(update_no, xml);
    }
    else
    {
      func(update_no, xml_out);
    }
  }









/*
static int local_site2(const int* coord, const int* latt_size)
  {
                int order = 0;
               for(int mmu=3; mmu >= 1; --mmu)
                        order = latt_size[mmu-1]*(coord[mmu] + order);
                order += coord[0];
                return order;
    }





static int
get_nodenum(const int *x, int *l,  int nd)
{
  int i, n;
  n = 0;
  for(i=nd-1; i>=0; i--) {
    int k=i;

        n = (n*l[k]) + x[k];
          }
            return n;
            }





*/




/*
class lattice_propagator
{
public:
int* site_vec;
complex<double> *A;
int* subgs;



lattice_propagator(LatticePropagator &chroma_propagator ,int *subgs1,int *site_vec1)
{
A=(complex<double>*)&(chroma_propagator.elem(0).elem(0,0).elem(0,0));
subgs=subgs1;
site_vec=site_vec1;
}

lattice_propagator(complex<double> *chroma_propagator,int *subgs1,int *site_vec1)
{
A=chroma_propagator;
subgs=subgs1;
site_vec=site_vec1;
}


complex<double> peeksite(int* site,int ii=0,int jj=0,int ll=0,int mm=0) //ii=spin_row jj=spin_rank ll=color_row mm=color_rank
{

int coords[4]={site[0]/subgs[0],site[1]/subgs[1],site[2]/subgs[2],site[3]/subgs[3]};
int N_sub[4]={site_vec[0]/subgs[0],site_vec[1]/subgs[1],site_vec[2]/subgs[2],site_vec[3]/subgs[3]};
//int rank = QMP_get_node_number(); //当前节点编号
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);


int nodenum=get_nodenum(coords,N_sub,4); //当前坐标节点编号


double dest[2];


if(rank==nodenum)
{
     int coord[4];
     for(int i=0;i<4;i++)
     {coord[i]=site[i];}
      int subgrid_vol_cb = (subgs[0]*subgs[1]*subgs[2]*subgs[3]) >> 1;

     int subgrid_cb_nrow[4];
     for(int i=0;i<4;i++)
     {subgrid_cb_nrow[i]=subgs[i];}

      subgrid_cb_nrow[0] >>= 1;

      int cb = 0;
      for(int m=0; m < 4; ++m)
        cb += coord[m];
      cb &= 1;
      int subgrid_cb_coord[4];
      subgrid_cb_coord[0] = (coord[0] >> 1) % subgrid_cb_nrow[0];
      for(int i=1; i < 4; ++i)
        subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];


     int t=local_site2(subgrid_cb_coord, subgrid_cb_nrow) + cb*subgrid_vol_cb;
      dest[0]= A[t*144+ii*36+jj*9+ll*3+mm].real();
      dest[1]= A[t*144+ii*36+jj*9+ll*3+mm].imag();

}





if(nodenum==0)
{
MPI_Bcast(dest,2,MPI_DOUBLE,0, MPI_COMM_WORLD);
complex<double> dest2(dest[0],dest[1]);
printf("finished a sendToWait node 0=%i \n",rank);
return dest2;
}
else
{
if(rank==nodenum)
{     

MPI_Send(dest,2,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
}
}

if(rank==0)
{
MPI_Recv(dest,2,MPI_DOUBLE,nodenum,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}


MPI_Bcast(dest,2,MPI_DOUBLE,0, MPI_COMM_WORLD);
printf("finished a sendToWait node=rank  %i \n",rank);
complex<double> dest2(dest[0],dest[1]);
return dest2;

}
};



*/





/*
class lattice_fermion
{
public:
vector<int> site_vec;
complex<double> *A;
lattice_fermion(LatticeFermion &chroma_fermi)
{
A=(complex<double>*)&(chroma_fermi.elem(0).elem(0).elem(0));
}

lattice_fermion(complex<double> *chroma_fermi)
{
A=chroma_fermi;
}


complex<double> peeksite(vector<int> site , vector<int> site_vec,int ii=0, int ll=0)//ii=spin ll=color
{

int length=site_vec[0]*site_vec[1]*site_vec[2]*site_vec[3];
int vol_cb;
if(length%2==0)
{vol_cb=(length)/2;}
if(length%2==1)
{vol_cb=(length-1)/2;}

int nx=site_vec[0];
int ny=site_vec[1];
int nz=site_vec[2];
int nt=site_vec[3];


int x=site[0];
int y=site[1];
int z=site[2];
int t=site[3];


int order=0;
int cb=x+y+z+t;

//判断nx的奇偶性
if(site_vec[0]%2==0)  
//{order = (nx/2)*(y+(ny*(z+nz*(t))));}
{order = t*nz*ny*nx/2+z*ny*nx/2+y*nx/2;}
if(site_vec[0]%2==1)
//{order = ((nx-1)/2)*(y+(ny*(z+nz*(t))));}
{order = t*nz*ny*(nx-1)/2+z*ny*(nx-1)/2+y*(nx-1)/2;}
//判断x奇偶性
if(x%2==0)
{x=(x/2);}
if(x%2==1)
{x=(x-1)/2;}

order+=x;
//判断x+y+z+t的奇偶性
cb &=1;
printf("vol_cb=%i\n",vol_cb);
return A[(order+cb*vol_cb)*12+ii*3+ll];
}
};
*/

/*

class lattice_gauge
{
public:
int* site_vec;
complex<double> *A[4];
int* subgs;
lattice_gauge(multi1d<LatticeColorMatrix> &chroma_gauge ,int *subgs1 ,int *site_vec1)
{
A[0]=(complex<double>*)&(chroma_gauge[0].elem(0).elem());
A[1]=(complex<double>*)&(chroma_gauge[1].elem(0).elem());
A[2]=(complex<double>*)&(chroma_gauge[2].elem(0).elem());
A[3]=(complex<double>*)&(chroma_gauge[3].elem(0).elem());
subgs=subgs1;
site_vec=site_vec1;
}

lattice_gauge(complex<double> *chroma_gauge[4],int *subgs1 ,int *site_vec1)
{
A[0]=chroma_gauge[0];
A[1]=chroma_gauge[1];
A[2]=chroma_gauge[2];
A[3]=chroma_gauge[3];
subgs=subgs1;
site_vec=site_vec1;
}



complex<double> peeksite(int* site, int ll=0,int mm=0,int dd=0)//ll=color_row mm=color_rank dd=dir
{

int coords[4]={site[0]/subgs[0],site[1]/subgs[1],site[2]/subgs[2],site[3]/subgs[3]};

int N_sub[4]={ site_vec[0]/subgs[0],site_vec[1]/subgs[1],site_vec[2]/subgs[2],site_vec[3]/subgs[3]  };




//int nodenum=(coords[3]*N_sub[2]*N_sub[1]*N_sub[0]+coords[2]*N_sub[1]*N_sub[0]+coords[1]*N_sub[0]+coords[0]);


//int rank = QMP_get_node_number(); //当前节点编号
int rank ;

MPI_Comm_rank (MPI_COMM_WORLD, &rank);

//int nodenum=QMP_get_node_number_from(coords); //当前坐标节点编号

int nodenum=get_nodenum(coords, N_sub, 4);



double dest[2];


if(rank==nodenum)
{
     int coord[4];
     for(int i=0;i<4;i++)
     {coord[i]=site[i];}
     int subgrid_vol_cb = (subgs[0]*subgs[1]*subgs[2]*subgs[3]) >> 1;
     int subgrid_cb_nrow[4];
     for(int i=0;i<4;i++)
     {subgrid_cb_nrow[i]=subgs[i];}

      subgrid_cb_nrow[0] >>= 1;

      int cb = 0;
      for(int m=0; m < 4; ++m)
        cb += coord[m];
      cb &= 1;


      int subgrid_cb_coord[4];
      subgrid_cb_coord[0] = (coord[0] >> 1) % subgrid_cb_nrow[0];
      for(int i=1; i < 4; ++i)
        subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];


     int t=local_site2(subgrid_cb_coord, subgrid_cb_nrow) + cb*subgrid_vol_cb;
      dest[0]= A[dd][t*9+ll*3+mm].real();
      dest[1]= A[dd][t*9+ll*3+mm].imag();

}



if(nodenum==0)
{
MPI_Bcast(dest,2,MPI_DOUBLE,0, MPI_COMM_WORLD);
complex<double> dest2(dest[0],dest[1]);
printf("finished a sendToWait_node=0 %i \n",rank);
return dest2;
}
else
{
if(rank==nodenum)
{
MPI_Send(dest,2,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
}
}

if(rank==0)
{
MPI_Recv(dest,2,MPI_DOUBLE,nodenum,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

}

MPI_Bcast(dest,2,MPI_DOUBLE,0, MPI_COMM_WORLD);
printf("finished a sendToWait node=rank  %i \n",rank);

complex<double> dest2(dest[0],dest[1]);
return dest2;

}
};


*/













  // Real work done here
  void 
  InlinePropagator2::func(unsigned long update_no,
			 XMLWriter& xml_out) 
  {
    START_CODE();

    QDPIO::cout << InlinePropagator2Env::name << ": propagator calculation" << std::endl;



//CHECK lattice_gauge and propagator

      LatticePropagator& source_tmp =
        TheNamedObjMap::Instance().getData<LatticePropagator>(params.named_obj.source_id);

    multi1d<LatticeColorMatrix>& u =
      TheNamedObjMap::Instance().getData< multi1d<LatticeColorMatrix> >(params.named_obj.gauge_id);

//u[0]=0;
//u[1]=0;
//u[2]=0;
//u[3]=0;


int *site_vec;
int site[4];


site[0]=1;
site[1]=3;
site[2]=2;
site[3]=50;

int spcl[4]={2,2,2,2};
int spd[3]={0,0,0};



site_vec=(int*)(&Layout::lattSize()[0]);  
int *subgs;
subgs=(int*)(&Layout::subgridLattSize()[0]);

cout<< subgs[0]<< "  " << subgs[1]<< "  " << subgs[2]<<"  "  << subgs[3]<<endl;



complex<double> *chroma_gauge[4];
chroma_gauge[0]=(complex<double>*)&(u[0].elem(0).elem());
chroma_gauge[1]=(complex<double>*)&(u[1].elem(0).elem());
chroma_gauge[2]=(complex<double>*)&(u[2].elem(0).elem());
chroma_gauge[3]=(complex<double>*)&(u[3].elem(0).elem());



lattice_gauge B(chroma_gauge,subgs,site_vec);

/*lattice_propagator A((complex<double>*)&(source_tmp.elem(0).elem(0,0).elem(0,0)),subgs,site_vec);
multi1d<int> site1;
site1.resize(4);


   site1[0] = site[0]; site1[1] = site[1]; site1[2] = site[2];site1[3] = site[3] ;    

   LatticeSpinMatrix tmp=peekColor(  source_tmp  , spcl[2], spcl[3]  )  ;
   LatticeComplex tmp1=peekSpin(tmp , spcl[0], spcl[1]);
  
   complex<double> a(peekSite(tmp1,site1).elem().elem().elem().real(),
                     peekSite(tmp1,site1).elem().elem().elem().imag());
QDPIO::cout << "chroma_peeksite_propagator="<< a.real() <<"  "<< a.imag() <<std::endl;

complex<double> T=A.peeksite(site,spcl[0],spcl[1],spcl[2],spcl[3]);
QDPIO::cout << "peeksite_propagator="<< T.real()<<"   "<< T.imag()  <<std::endl;


   LatticeComplex tmp2=peekColor(u[0] , spd[0], spd[1]);
   complex<double> b(peekSite(tmp2,site1).elem().elem().elem().real(),
                     peekSite(tmp2,site1).elem().elem().elem().imag());
QDPIO::cout << "chroma_peeksite_gauge="<< b.real() <<"  "<< b.imag() <<std::endl;

complex<double> R=B.peeksite(site,spd[0],spd[1],0);
QDPIO::cout << "peeksite_gauge="<<R.real()<<"    "<< R.imag() <<std::endl;


*/


                                      //CHROMA DSLASH
//////////////////////////////////////////////////////////////////////////////////
    std::istringstream  xml_s(params.param.fermact.xml);
    XMLReader  fermacttop(xml_s);
{



        typedef LatticeFermion               T;
        typedef multi1d<LatticeColorMatrix>  P;
        typedef multi1d<LatticeColorMatrix>  Q;

        Handle<FermAct4D<T,P,Q> > S_f1(nullptr);
          S_f1=dynamic_cast<FermAct4D<T,P,Q>*>(TheFermionActionFactory::Instance().createObject(params.param.fermact.id,
                                       fermacttop,
                                       params.param.fermact.path));


        LatticeFermion dest1;
        LatticeFermion tmp_F;
        LatticeFermion tmp_F_test;
        Handle< FermState<T,P,Q> > state(S_f1->createState(u));
        PropToFerm(source_tmp,tmp_F,0,0);
        (*(S_f1 ->linOp(state)))(dest1,tmp_F,PLUS);
        (*(S_f1 ->linOp(state)))(tmp_F_test,dest1,PLUS);
        dest1=zero;
        (*(S_f1 ->linOp(state)))(dest1,tmp_F_test,PLUS);
        tmp_F_test=zero;
        (*(S_f1 ->linOp(state)))(tmp_F_test,dest1,MINUS);



//        QDPIO::cout << "chroma_dslash_norm2_2:"<<norm2(tmp_F_test)<< std::endl;
        


 
                                    //CHECK EOPRICONDITION  DSLASH
////////////////////////////////////////////////////////////////////////////////////

LatticeFermion dest2;
LatticeFermion dest3;
LatticeFermion dest4;
LatticeFermion dest5;
LatticeFermion dest6;
LatticeFermion dest7;
lattice_fermion src((complex<double>*)&(dest1.elem(0).elem(0).elem(0)),subgs,site_vec);
//lattice_fermion dest_OO((complex<double>*)&(dest2.elem(0).elem(0).elem(0)),subgs,site_vec);
//lattice_fermion dest_EE((complex<double>*)&(dest3.elem(0).elem(0).elem(0)),subgs,site_vec);
lattice_fermion dest_EO((complex<double>*)&(dest4.elem(0).elem(0).elem(0)),subgs,site_vec);
lattice_fermion dest_OE((complex<double>*)&(dest5.elem(0).elem(0).elem(0)),subgs,site_vec);
lattice_fermion dest_chroma((complex<double>*)&(tmp_F_test.elem(0).elem(0).elem(0)),subgs,site_vec);
//lattice_fermion dest_test((complex<double>*)&(dest6.elem(0).elem(0).elem(0)),subgs,site_vec);
//lattice_fermion dest_test1((complex<double>*)&(dest7.elem(0).elem(0).elem(0)),subgs,site_vec);

//DslashOO(src,dest_OO,1);


/*for(int i=0;i<src.size;i++)
{
if(dest_OO.A[i].real()!=0)
{printf("dslash_OO=%f\n",dest_OO.A[i].real());}
}*/
//printf("nrom_2(DslashOO_1)=%.10f\n",norm_2(dest_OO));
//DslashEE(src,dest_EE,1);
//printf("nrom_2(DslashEE_1)=%.10f\n",norm_2(dest_EE));



//DslashEO(src,dest_EO,B,false);


//DslashOE(src,dest_OE,B,false);

Dslashoffd(src,dest_EO,B,true,0);
Dslashoffd(src,dest_OE,B,true,1);


//for(int i=0;i<dest.size;i++)
//{printf("DslashEO=%f\n",dest.A[i]);}


/*
//printf("src=%f\n",norm_2(src));
printf("nrom_2(chroma_dslash)=%f\n",norm_2(dest_chroma));
printf("nrom_2(DslashEO)=%f\n",norm_2(dest_EO));
printf("nrom_2(DslashOE)=%f\n",norm_2(dest_OE));
printf("nrom_2(DslashOO)=%f\n",norm_2(dest_OO));
printf("nrom_2(DslashEE)=%f\n",norm_2(dest_EE));


int cc= subgs[0]*subgs[1]*subgs[2]*subgs[3]/2*12;

cout<< subgs[0]<< "  " << subgs[1]<< "  " << subgs[2]<<"  "  << subgs[3]<<endl;
*/
/*
for(int i=0;i<cc;i=i+12)
{

//if(dest_chroma.A[i].real()!=0 or dest_chroma.A[cc+i].real()!=0 or dest_OE.A[i+cc].real()!=0 or dest_EO.A[i].real()!=0)
{
//if(dest_chroma.A[i].real()+ dest_EO.A[i].real()!=0)
//if(dest_OE.A[i+cc].real()!=0 or dest_EO.A[i].real()!=0)
{
printf("i=%i\n",i/12); 
printf("dslashE_chroma=%f %f\n",dest_chroma.A[i].real(),dest_chroma.A[i].imag());
printf("DslashE=%f %f\n",dest_EO.A[i].real(),dest_EO.A[i].imag());
printf("DslashO=%f %f\n",dest_OE.A[i+cc].real(),dest_OE.A[i+cc].imag());
printf("dslashO_chroma=%f %f\n",dest_chroma.A[cc+i].real(),dest_chroma.A[cc+i].imag());
}
}
}
*/




//printf("nrom_2(dslashO)=%f\n",norm_2_O(dest_chroma-dest_OE));


//printf("nrom_2(chroma_dslash-dslash_test)=%.12f\n",norm_2((((dest_chroma-dest_EE)-dest_OO)-dest_EO)-dest_OE));
printf("nrom_2(chroma_dslash-dslash_test)=%.12f\n",norm_2((dest_chroma-dest_EO)-dest_OE));





}




    END_CODE();
  } 

}
