// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <iostream>
#include <gunrock/app/bc/bc_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

/******************************************************************************
* Main
******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct
{
    /**
     * @brief the actual main function, after type switching
     * @tparam VertexT    Type of vertex identifier
     * @tparam SizeT      Type of graph size, i.e. type of edge identifier
     * @tparam ValueT     Type of edge values
     * @param  parameters Command line parameters
     * @param  v,s,val    Place holders for type deduction
     * \return cudaError_t error message(s), if any
     */
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_EDGE_VALUES | graph::HAS_CSR> GraphT; // PR different
        
        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph;

        parameters.Set("undirected", true);

        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());

        auto &graph_coo = graph.coo();
        for(VertexT i=0; i < graph_coo.edges; i++) {
            auto &edge_pair = graph_coo.edge_pairs[i];
            std::cout << edge_pair.x << "|" << edge_pair.y << std::endl;
        }
        
        // GUARD_CU(app::Set_Srcs    (parameters, graph));
        // int num_srcs = 0;

        // TODO: reference result on CPU, e.e.:
        // ValueT  **ref_distances = NULL;
        bool quick = parameters.Get<bool>("quick");
        bool quiet = parameters.Get<bool>("quiet");
        
        if (!quick)
        {
            // std::string validation = parameters.Get<std::string>("validation");
            util::PrintMsg("Computing reference value ...", !quiet);

            // // TODO: get srcs if needed, e.g.:
            // std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT> >("srcs");
            // num_srcs = srcs.size();

            // SizeT nodes = graph.nodes;
            // // TODO: problem specific data, e.g.:
            // ref_distances = new ValueT*[num_srcs];
            // for (int i = 0; i < num_srcs; i++)
            // {
            //     // ref_distances[i] = new ValueT[nodes];
            //     // VertexT src = srcs[i];
            //     util::PrintMsg("__________________________", !quiet);
            //     float elapsed = app::Template::CPU_Reference(
            //         graph.csr(),
            //         // TODO: add problem specific data, e.g.:
            //         // ref_distances[i], NULL, src,
            //         quiet);
            //     util::PrintMsg("--------------------------\nRun "
            //         + std::to_string(i) + " elapsed: "
            //         + std::to_string(elapsed)
            //         //+ " ms, src = " + std::to_string(src)
            //         , !quiet);
            // }
        }

//         // TODO: add other switching parameters, if needed
//         std::vector<std::string> switches{"advance-mode"};
//         // TODO: add problem specific data
//         GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
//             [/*ref_distances*/](util::Parameters &parameters, GraphT &graph)
//             {
//                 return app::Template::RunTests(parameters, graph/*, ref_distances*/);
//             }));

//         if (!quick)
//         {
//             // TODO: deallocate host references, e.g.:
//             // for (int i = 0; i < num_srcs; i ++)
//             // {
//             //    delete[] ref_distances[i]; ref_distances[i] = NULL;
//             // }
//             // delete[] ref_distances; ref_distances = NULL;
//         }
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test bc");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::bc::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    // TODO: change available graph types, according to requirements
    return app::Switch_Types<
        app::VERTEXT_U32B | // app::VERTEXT_U64B |
        app::SIZET_U32B | // app::SIZET_U64B |
        app::VALUET_F32B | 
        app::UNDIRECTED>
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
