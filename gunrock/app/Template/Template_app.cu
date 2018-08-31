// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file Template_app.cu
 *
 * @brief single-source shortest path (SSSP) application
 */

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

#include <gunrock/app/Template/Template_enactor.cuh>
#include <gunrock/app/Template/Template_test.cuh>

namespace gunrock {
namespace app {
namespace Template {

cudaError_t UseParameters(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(UseParameters_app    (parameters));
    GUARD_CU(UseParameters_problem(parameters));
    GUARD_CU(UseParameters_enactor(parameters));

    // <todo>
    // GUARD_CU(parameters.Use<std::string>(
    //    "src",
    //    util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    "0",
    //    "<Vertex-ID|random|largestdegree> The source vertices\n"
    //    "\tIf random, randomly select non-zero degree vertices;\n"
    //    "\tIf largestdegree, select vertices with largest degrees",
    //    __FILE__, __LINE__));
    // </todo>

    return retval;
}

struct RunParameters {

    // <todo> declare parameters, eg:
    // VertexT* srcs;
    // </todo>

    RunParameters() {}

    cudaError_t Init(util::Parameters &parameters) {
        cudaError_t retval = cudaSuccess;
        
        // <todo> set parameters
        // </todo>
        
        return retval;
    }
    
    cudaError_t SetRun(int run_num) {
        // This could do something like increment the srcs
    }
    
    cudaError_t Release() {
        // Release parameters
    }
};

// ========================================= No todos below here =============================================

template <
    typename GraphT, 
    typename ValueT = typename GraphT::ValueT,
    typename VertexT = typename GraphT::VertexT,
    typename SizeT = typename GraphT::SizeT,
    typename ResultT
>
cudaError_t Run(
    util::Parameters &parameters,
    GraphT           &graph,
    ResultT          &result,
    util::Location target = util::DEVICE)
{
    cudaError_t retval = cudaSuccess;
    
    typedef Problem<GraphT, RunParameters, ResultT> ProblemT;
    typedef Enactor<ProblemT> EnactorT;
    
    util::CpuTimer cpu_timer, total_timer;
    cpu_timer.Start(); total_timer.Start();
    
    // Init info
    util::Info info("Template", parameters, graph);

    // Problem specific parameters
    RunParameters run_parameters;
    run_parameters.Init(parameters);
    
    // Init problem
    ProblemT problem(parameters);
    GUARD_CU(problem.Init(graph, target));
    
    // Init enactor
    EnactorT enactor;
    GUARD_CU(enactor.Init(problem, target));
    
    cpu_timer.Stop();
    parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());
    
    int num_runs = parameters.Get<int >("num-runs");
    for (int run_num = 0; run_num < num_runs; ++run_num) {
        run_parameters.SetRun(run_num);
        
        GUARD_CU(problem.Reset(run_parameters, target));
        GUARD_CU(enactor.Reset(/*run_parameters,*/ target));
        
        cpu_timer.Start();
        GUARD_CU(enactor.Enact(/*run_parameters*/));
        cpu_timer.Stop();
        info.CollectSingleRun(cpu_timer.ElapsedMillis());
    }

    cpu_timer.Start();
    GUARD_CU(problem.Extract(result));

    // <todo> compute running statistics
    // !! This might be broken now
    // TODO: change NULL to problem specific per-vertex visited marker, e.g. h_distances
    // info.ComputeTraversalStats(enactor, (VertexT*)NULL);
    //Display_Memory_Usage(problem);
    // #ifdef ENABLE_PERFORMANCE_PROFILING
        //Display_Performance_Profiling(enactor);
    // #endif

    // Clean up
    run_parameters.Release();
    GUARD_CU(enactor.Release(target));
    GUARD_CU(problem.Release(target));
    cpu_timer.Stop(); total_timer.Stop();

    info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
    return retval;
}

} // namespace Template
} // namespace app
} // namespace gunrock
