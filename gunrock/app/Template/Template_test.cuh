#pragma once

namespace gunrock {
namespace app {
namespace Template {

template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT,
    typename VertexT = typename GraphT::VertexT,
    typename SizeT = typename GraphT::SizeT
>
struct Result {
    
    // <todo> declare data structures
    SizeT num_nodes;
    ValueT *degrees;
    // </todo>
    
    Result() {}
    
    cudaError_t Init(GraphT &graph, util::Parameters &parameters) {
        cudaError_t retval = cudaSuccess;
        
        // <todo> initialize data structures
        num_nodes = graph.nodes;
        degrees = new ValueT[num_nodes];
        // </todo>
        
        return retval;
    }

    cudaError_t Release() {
        cudaError_t retval = cudaSuccess;

        // <todo> release data structures
        delete[] degrees; degrees = NULL;
        // </todo>
        
        return retval;
    }
    
    cudaError_t Reset() {
        for(SizeT v = 0; v < num_nodes; ++v) {
            degrees[v] = 0;
        }
    }
    
};

template <
    typename GraphT, 
    typename ValueT = typename GraphT::ValueT,
    typename VertexT = typename GraphT::VertexT,
    typename SizeT = typename GraphT::SizeT,
    typename ResultT
>
double CPU_Reference(const GraphT &graph, ResultT &result, bool quiet) {
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <todo> 
    // implement reference implementation
    for(SizeT v = 0; v < graph.nodes; ++v) {
        result.degrees[v] = graph.row_offsets[v + 1] - graph.row_offsets[v];
    }
    // </todo>

    // <debug>
    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("%d -> %d\n", v, result.degrees[v]);
    }
    // </debug>
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;

}

template <typename ResultT>
int Validate_Results(const ResultT &result1, const ResultT &result2) {
    
    int num_errors = 0;
    
    printf("validate results\n");
    
    for(int i = 0; i < 10; i++) {
        printf("%d %d\n", result1.degrees[i], result2.degrees[i]);
    }
    
    // // <todo> add code that compares two result structs
    // int num_nodes = sizeof(result1.degrees) / sizeof(result1.degrees[0]);
    // for(int v = 0; v < num_nodes; ++v) {
    //     // <debug>
    //     printf("%d %d\n", result1.degrees[v], result2.degrees[v]);
    //     // </debug>
        
    //     if(result1.degrees[v] != result2.degrees[v]) {
    //         num_errors++;
    //         printf("result1.degrees[%d] != result2.degrees[%d]", v, v);
    //     }
    // }
    // // </todo>

    return num_errors;
}

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
