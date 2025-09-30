# P2P Connector
P2P connector is used for testing the afd implementation for deepseek-v2-lite models. It uses torch.distributed to send/recv intermediate tensors between attn and ffn instances. 

1. Attn

```
vllm serve "/path/to/DeepSeek-V2-Lite"  --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager  --afd-config '{"afd_connector":"p2pconnector", "afd_role": "attention", "afd_host":"127.0.0.1", "afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

```

2. FFN

```
vllm fserver "/path/to/DeepSeek-V2-Lite" --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"1", "afd_role": "ffn", "afd_host":"127.0.0.1", "afd_port":"29500", "afd_extra_config":{"afd_size":"2A2F"}}'
```

