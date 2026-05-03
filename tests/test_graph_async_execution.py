import inspect

from src.graph import nodes


def test_io_bound_nodes_are_async_functions():
    assert inspect.iscoroutinefunction(nodes.search_and_memory_node)
    assert inspect.iscoroutinefunction(nodes.memory_context_node)
    assert inspect.iscoroutinefunction(nodes.rerank_node)
    assert inspect.iscoroutinefunction(nodes.summarize_node)
    assert inspect.iscoroutinefunction(nodes.report_node)
