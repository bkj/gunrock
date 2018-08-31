// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

#include <gunrock/app/Template/Template_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

// <todo>
namespace APP_NAMESPACE = app::Template;

const auto GRAPH_TYPES = graph::HAS_EDGE_VALUES | graph::HAS_CSR;

const auto SWITCH_TYPES = app::VERTEXT_U32B | app::VERTEXT_U64B |
        app::SIZET_U32B | app::SIZET_U64B |
        app::VALUET_U32B | app::DIRECTED | app::UNDIRECTED;

const bool SET_SRCS = false;

std::vector<std::string> switches{"advance-mode"};
// </todo>

// =============== No todos below this line ================

struct main_struct {
    template <typename VertexT, typename SizeT, typename ValueT>
    cudaError_t operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
        bool quick = parameters.Get<bool>("quick");
        bool quiet = parameters.Get<bool>("quiet");
        
        typedef typename app::TestGraph<VertexT, SizeT, ValueT, GRAPH_TYPES> GraphT;
        typedef typename APP_NAMESPACE::Result<GraphT> ResultT;
        
        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph;

        // Load graph
        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());
        
        if(SET_SRCS) {
            GUARD_CU(app::Set_Srcs(parameters, graph));
        }
        
        ResultT reference_results;
        if (!quick) {
            reference_results.Init(graph, parameters);
            reference_results.Reset();
            APP_NAMESPACE::CPU_Reference(graph, reference_results, quiet);
        }
            
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [reference_results, quick](util::Parameters &parameters, GraphT &graph) {
                cudaError_t retval = cudaSuccess;
                
                ResultT gunrock_results;
                gunrock_results.Init(graph, parameters);
                gunrock_results.Reset();
                
                APP_NAMESPACE::Run(parameters, graph, gunrock_results);
                
                if(!quick) {
                    APP_NAMESPACE::Validate_Results(gunrock_results, reference_results);   
                }
                
                gunrock_results.Release();
                return retval;
            }));
        
        reference_results.Release();
        return retval;
    }
};

int main(int argc, char** argv) {
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test Template");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(APP_NAMESPACE::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help")) {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());
    
    return app::Switch_Types<SWITCH_TYPES>(parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
