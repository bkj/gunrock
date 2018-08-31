#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace Template {

cudaError_t UseParameters_problem(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(gunrock::app::UseParameters_problem(parameters));

    // <todo> Add problem specific command-line parameter usages here, e.g.:
    // GUARD_CU(parameters.Use<bool>(
    //    "mark-pred",
    //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    false,
    //    "Whether to mark predecessor info.",
    //    __FILE__, __LINE__));
    // </todo>

    return retval;
}

template <
    typename _GraphT,
    typename RunParametersT,
    typename ResultT,
    ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG>
{
    typedef _GraphT GraphT;
    static const ProblemFlag FLAG = _FLAG;

    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;
    
    struct DataSlice : BaseDataSlice
    {
        // <todo> add problem specfic data structures
        util::Array1D<SizeT, ValueT> degrees;
        util::Array1D<SizeT, ValueT> seen;
        // </todo>
        
        DataSlice() : BaseDataSlice() {
            // <todo> Set names of the problem specific arrays
            degrees.SetName("degrees");
            seen.SetName("seen");
            // </todo>
        }
        
        virtual ~DataSlice() { Release(); }
        
        cudaError_t Release(util::Location target = util::LOCATION_ALL)
        {
            cudaError_t retval = cudaSuccess;
            if (target & util::DEVICE) {
                GUARD_CU(util::SetDevice(this->gpu_idx));
            }

            // <todo> Release problem specific data, e.g.:
            GUARD_CU(degrees.Release(target));
            GUARD_CU(seen.Release(target));
            // </todo>

            GUARD_CU(BaseDataSlice::Release(target));
            return retval;
        }
        
        cudaError_t Init(
            GraphT        &sub_graph,
            int            num_gpus = 1,
            int            gpu_idx = 0,
            util::Location target  = util::DEVICE,
            ProblemFlag    flag    = Problem_None)
        {
            cudaError_t retval  = cudaSuccess;
            
            GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

            // <todo> allocate problem specific data here, e.g.:
            GUARD_CU(degrees.Allocate(sub_graph.nodes, target));
            GUARD_CU(seen.Allocate(sub_graph.nodes, target));
            // </todo>

            if (target & util::DEVICE) {
                GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this -> stream));
            }
            return retval;
        }


        cudaError_t Reset(util::Location target = util::DEVICE)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = this -> sub_graph -> nodes;

            // <todo> ensure size of problem specific data, e.g.:
            GUARD_CU(degrees.EnsureSize_(nodes, target));
            GUARD_CU(seen.EnsureSize_(nodes, target));
            // </todo>

            // <todo> Reset data
            GUARD_CU(degrees.ForEach([]__host__ __device__ (ValueT &x){
               x = 0;
            }, nodes, target, this -> stream));
            
            GUARD_CU(seen.ForEach([]__host__ __device__ (ValueT &x){
               x = 0;
            }, nodes, target, this -> stream));
            // </todo>
            
            return retval;
        }
    }; // DataSlice

    // Members
    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // Methods

    /**
     * @brief SSSPProblem default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL)
    {
    }

    /**
     * @brief SSSPProblem default destructor
     */
    virtual ~Problem()
    {
        Release();
    }
    
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices == NULL) return retval;
        for (int i = 0; i < this->num_gpus; i++)
            GUARD_CU(data_slices[i].Release(target));

        if ((target & util::HOST) != 0 &&
            data_slices[0].GetPointer(util::DEVICE) == NULL)
        {
            delete[] data_slices; data_slices=NULL;
        }
        GUARD_CU(BaseProblem::Release(target));
        return retval;
    }
    
    
    cudaError_t Extract(
        ResultT &result,
        util::Location  target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        SizeT nodes = this -> org_graph -> nodes;

        if (this-> num_gpus == 1) {
            auto &data_slice = data_slices[0][0];

            // Set device
            if (target == util::DEVICE) {
                GUARD_CU(util::SetDevice(this->gpu_idx[0]));
                
                // <todo>: extract the results from single GPU, e.g.:
                GUARD_CU(data_slice.degrees.SetPointer(result.degrees, nodes, util::HOST));
                GUARD_CU(data_slice.degrees.Move(util::DEVICE, util::HOST));
                // </todo>
                
            } else if (target == util::HOST) {
                // <todo>: extract the results from single CPU, e.g.:
                GUARD_CU(data_slice.degrees.ForEach(result.degrees,
                   []__host__ __device__
                   (const ValueT &degree, ValueT &h_degree){
                       h_degree = degree;
                   }, nodes, util::HOST));
                // </todo>
            }
        } else {
            // INCOMPLETE
        }

        return retval;
    }
    
    cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseProblem::Init(graph, target));
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        // <todo> get problem specific flags from parameters, e.g.:
        // if (this -> parameters.template Get<bool>("mark-pred"))
        //    this -> flag = this -> flag | Mark_Predecessors;
        // </todo>

        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
            data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

            GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

            auto &data_slice = data_slices[gpu][0];
            GUARD_CU(data_slice.Init(
                this -> sub_graphs[gpu],
                this -> num_gpus,
                this -> gpu_idx[gpu], 
                target,
                this -> flag
            ));
        }

        return retval;
    }
    
    cudaError_t Reset(RunParametersT run_parameters, util::Location target = util::DEVICE) {
        cudaError_t retval = cudaSuccess;

        // Reset all data slices
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            if (target & util::DEVICE) {
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            }
            GUARD_CU(data_slices[gpu]->Reset(target));
            GUARD_CU(data_slices[gpu].Move(util::HOST, target));
        }

        // <todo> Additional problem specific initialization
        // </todo>
        return retval;
    }

    /** @} */
};

} //namespace Template
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
