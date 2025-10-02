"""
Test script to verify the PPO bidexhands setup is working correctly.
This script performs basic tests on all components.
"""

import sys
import os
import numpy as np
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import mujoco as mj
        print("✓ MuJoCo imported successfully")
    except ImportError as e:
        print(f"✗ MuJoCo import failed: {e}")
        return False
    
    try:
        from primitive_controller.ppo import PPOAgent, PPOMemory
        print("✓ PPO modules imported successfully")
    except ImportError as e:
        print(f"✗ PPO import failed: {e}")
        return False
    
    try:
        from environment.rubiks_cube import RubiksCubeEnvironment
        print("✓ Environment module imported successfully")
    except ImportError as e:
        print(f"✗ Environment import failed: {e}")
        return False
    
    return True


def test_xml_files():
    """Test that XML files exist and are valid."""
    print("\nTesting XML files...")
    
    xml_files = [
        "bidexhands.xml",
        "left_hand.xml", 
        "right_hand.xml",
        "cube_rad.xml"
    ]
    
    for xml_file in xml_files:
        if os.path.exists(xml_file):
            print(f"✓ {xml_file} exists")
            try:
                # Try to load with MuJoCo
                model = mj.MjModel.from_xml_path(xml_file)
                print(f"✓ {xml_file} is valid MuJoCo XML")
            except Exception as e:
                print(f"✗ {xml_file} is invalid: {e}")
                return False
        else:
            print(f"✗ {xml_file} not found")
            return False
    
    return True


def test_ppo_agent():
    """Test PPO agent creation and basic functionality."""
    print("\nTesting PPO agent...")
    
    try:
        # Create agent
        agent = PPOAgent(
            state_dim=50,
            action_dim=20,
            lr=1e-3,
            device='cpu'
        )
        print("✓ PPO agent created successfully")
        
        # Test action selection
        state = np.random.randn(50)
        action, log_prob, value = agent.select_action(state)
        
        assert action.shape == (20,), f"Action shape {action.shape} != (20,)"
        assert log_prob.shape == (1,), f"Log prob shape {log_prob.shape} != (1,)"
        assert value.shape == (1,), f"Value shape {value.shape} != (1,)"
        print("✓ Action selection works correctly")
        
        # Test memory
        memory = PPOMemory()
        memory.store(state, action, reward=1.0, next_state=state, done=False, log_prob=log_prob, value=value)
        print("✓ Memory storage works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ PPO agent test failed: {e}")
        return False


def test_environment():
    """Test environment creation and basic functionality."""
    print("\nTesting environment...")
    
    try:
        # Create environment
        env = RubiksCubeEnvironment(
            xml_path="bidexhands.xml",
            max_episode_steps=100,
            enable_viewer=False
        )
        print("✓ Environment created successfully")
        
        # Test initialization
        state = env.initialize()
        assert isinstance(state, np.ndarray), "State should be numpy array"
        assert state.dtype == np.float32, "State should be float32"
        print("✓ Environment initialization works")
        
        # Test action parsing
        action = np.random.uniform(-0.1, 0.1, env.action_dim)
        parsed_actions = env.get_action(action)
        assert 'left_hand' in parsed_actions, "Should have left_hand actions"
        assert 'right_hand' in parsed_actions, "Should have right_hand actions"
        assert 'cube' in parsed_actions, "Should have cube actions"
        print("✓ Action parsing works correctly")
        
        # Test step
        next_state, reward, done, info = env.take_step(action)
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        print("✓ Environment step works correctly")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_training_script():
    """Test that training script can be imported and has required functions."""
    print("\nTesting training script...")
    
    try:
        # Import training script
        import train_bidexhands_ppo
        print("✓ Training script imported successfully")
        
        # Check for required functions
        assert hasattr(train_bidexhands_ppo, 'main'), "Training script should have main function"
        assert hasattr(train_bidexhands_ppo, 'parse_args'), "Training script should have parse_args function"
        print("✓ Training script has required functions")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        return False


def test_device_compatibility():
    """Test device compatibility."""
    print("\nTesting device compatibility...")
    
    try:
        # Test CPU
        agent_cpu = PPOAgent(10, 5, device='cpu')
        state = np.random.randn(10)
        action, _, _ = agent_cpu.select_action(state)
        print("✓ CPU device works correctly")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            agent_cuda = PPOAgent(10, 5, device='cuda')
            action, _, _ = agent_cuda.select_action(state)
            print("✓ CUDA device works correctly")
        else:
            print("⚠ CUDA not available (this is normal)")
        
        return True
        
    except Exception as e:
        print(f"✗ Device compatibility test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("PPO Bidexhands Setup Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("XML Files", test_xml_files),
        ("PPO Agent", test_ppo_agent),
        ("Environment", test_environment),
        ("Training Script", test_training_script),
        ("Device Compatibility", test_device_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready for training.")
        print("\nTo start training, run:")
        print("python train_bidexhands_ppo.py --num_episodes 100 --enable_viewer")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed and XML files are present.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

