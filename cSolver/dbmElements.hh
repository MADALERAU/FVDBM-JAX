#include "Eigen/Dense"
#include <vector>
#include <iostream>

using Eigen::Vector;
using Eigen::RowVector;
using Eigen::Matrix;
using Eigen::Dynamic;

// Defining D2Q9 Dynamics
static const Matrix<double,2,9> ksi = (Matrix<double,2,9>() << 0, 1, 0, -1, 0, 1, -1, -1, 1,0, 0, 1, 0, -1, 1, 1, -1, -1).finished();
static const RowVector<double,9> w = (RowVector<double,9>() << 4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.).finished();

// Eq PDF Calculator
Vector<double,9> calcEq(double rho,Vector<double,2> vel){
    return (w.transpose().array()*rho*( 1 + 3*(ksi.transpose()*vel).array()+9/2*((ksi.transpose()*vel)).array().square() - 3/2*(vel.transpose()*vel).array().value())).matrix();
}


// General Element (Cell, Face, Node) 
class GeneralElement{
    public:
        Vector<double,9> pdf;
        double rho;
        Vector<double,2> vel;
        
        // CONSTRUCTOR
        GeneralElement(Vector<double,9>tempPdf,double tempRho=0,Vector<double,2> tempVel = Vector<double,2>{{0,0}}){
            pdf = tempPdf;      // Current PDF of element
            rho = tempRho;      // Current Density of element
            vel = tempVel;      // Current Velocity of element
        }
        
        // Calculates density & velocity
        void calcMacro(){
            rho = pdf.sum();
            vel = ksi*pdf/rho;
        }
};

class Cell: public GeneralElement {
    public:
        Vector<double,9> pdfEq;     // Equilibrium PDF 
        std::vector<GeneralElement*> faces;   // Faces which defines Cell


        // CONSTRUCTOR
        Cell(Vector<double,9>tempPdf,double tempRho=0,Vector<double,2> tempVel = Vector<double,2>{{0,0}})
            :GeneralElement(tempPdf,tempRho,tempVel){
        }

        void setEq(){           // updates Eq PDF
            calcMacro();
            pdfEq = calcEq(rho,vel);
        }

        Vector<double,9> getNeq(){      // Returns non Eq PDF
            return pdf-pdfEq;
        }
};

class Face: public GeneralElement {
    public:
        std::vector<GeneralElement*> nodes;           // Nodes which defines Face
        std::vector<GeneralElement*> stencilCells;    // Cells for stenciling
        Vector<double,Dynamic> stensilDists;   // Distance of Cells above

        void updatePDF(){       // Updates 

        }

        void updateGhosts(){    // identifies & Updates any ghost cells

        }

};

class Node: public GeneralElement {
    public:
        std::vector<Cell*> cells;       // Cells connected to Node
        Vector<double,Dynamic> cellDists;  // Distance of cells above

        // CONSTRUTOR
        Node(Vector<double,9>tempPdf,double tempRho=0,Vector<double,2> tempVel = Vector<double,2>{{0,0}})
            :GeneralElement(tempPdf,tempRho,tempVel){
        }

        void addCell(){
            
        }

        void updatePDF(){
            
            pdf = calcNeq()+calcBoundaryEq();
        }
        
        Vector<double,9> calcNeq(){
            Vector<double,9> neqPDF = Vector<double,9>::Zero();
            double denom = 0;
            for (int i = 0;i<cellDists.size();i++){
                neqPDF += cells[i]->getNeq()/cellDists[i];
                denom += 1/cellDists[i];
            }
            return neqPDF/denom;
        }

        Vector<double,9> calcBoundaryEq(){      // Placeholder for Boundary EQ calculation
            std::cout << "boundary messed up" << std::endl;
        }
};

class DirichetNode : public Node {
    public:
        double targ_rho = -1;
        Vector<double,2> targ_vel {0,0};

        Vector<double,9> calcBoundaryEq(){  // Direchet Boundary Equlibrium Claculation
            
            if (targ_rho > 0){      // if rho > 0; use density boundary
                vel = calcVel();
                return calcEq(targ_rho,vel);
            }
            else {      // velocity boundary
                rho = calcDensity();
                return calcEq(rho,targ_vel);
            }
        }

        Vector<double,2> calcVel(){     //extrapolates velocity from nearby cells
            Vector<double,2> avgVel = Vector<double,2>::Zero();
            double denom = 0;
            for (int i = 0;i<cellDists.size();i++){
                avgVel += cells[i]->vel/cellDists[i];
                denom += 1/cellDists[i];
            }
            return avgVel/denom;
        }

        double calcDensity(){       //extrapolates 
            double avgDensity = 0;
            double denom = 0;
            for (int i = 0;i<cellDists.size();i++){
                avgDensity += cells[i]->rho/cellDists[i];
                denom += 1/cellDists[i];
            }
            return avgDensity/denom;
        }

    };
